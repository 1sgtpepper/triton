#include "Dialect/NVWS/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/hopper/lib/Transforms/WarpSpecialization/CodePartitionUtility.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Tools/Sys/Dump.h"

#define DEBUG_TYPE "nvgpu-warp-specialization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

void doTaskPartition(triton::FuncOp &funcOp, unsigned numWarpGroups);
int doTaskIdPropagate(triton::FuncOp &funcOp);
bool doDataPartition(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers);
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups);

#define GEN_PASS_DEF_NVGPUWARPSPECIALIZATION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUWarpSpecializationPass
    : public impl::NVGPUWarpSpecializationBase<NVGPUWarpSpecializationPass> {
public:
  using impl::NVGPUWarpSpecializationBase<
      NVGPUWarpSpecializationPass>::NVGPUWarpSpecializationBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    SmallVector<scf::ForOp> loops;
    funcOp->walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(mlir::triton::kWarpSpecializeAttrName) &&
          triton::getNumStagesOrDefault(forOp, numStages) > 1)
        loops.push_back(forOp);
    });
    if (loops.empty())
      return;

    int numWarps = mlir::triton::gpu::lookupNumWarps(funcOp);
    if (numWarps != 4)
      return;

    // FIXME: skip warpspec if there is else block. Need to improve
    // CodePartitioning to correctly handle channels in else block.
    bool hasElse = false;
    funcOp->walk([&](scf::IfOp ifOp) {
      if (ifOp.elseBlock()) {
        hasElse = true;
      }
    });
    if (hasElse)
      return;

    bool hasPreexistingTaskIds = false;
    funcOp.walk([&](Operation *op) {
      hasPreexistingTaskIds |= op->hasAttr("async_task_id");
    });

    OpBuilder builder(funcOp);
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    unsigned numWarpGroups = 3;
    // FIXME: skip data partitioning with on-host TMA.
    bool success = false;
    for (; numWarpGroups >= 2; numWarpGroups--) {
      // Partition key ops into multiple async tasks.
      doTaskPartition(funcOp, numWarpGroups);
      if (dumpIntermediateSteps) {
        ::mlir::triton::tools::mlirDumpsOrDbgs()
            << "// -----// WarpSpec internal IR Dump After: doTaskPartition\n"
            << moduleOp << "\n\n\n";
      }
      // Propagate taskId.
      int retCode = doTaskIdPropagate(funcOp);
      if (retCode == -1)
        continue;
      if (dumpIntermediateSteps) {
        ::mlir::triton::tools::mlirDumpsOrDbgs()
            << "// -----// WarpSpec internal IR Dump After: doTaskIdPropagate\n"
            << moduleOp << "\n\n\n";
      }

      bool hasUnsupportedGather = false;
      bool hasUnsupportedAtomic = false;
      // Data partition follows loop initial values and yields, and rewrites the
      // function as a whole. Keep this preflight function-local rather than
      // duplicating the partitioner's dimension-aware slice analysis here.
      funcOp.walk([&](Operation *op) {
        if (isa<triton::GatherOp>(op)) {
          hasUnsupportedGather |= !op->getResult(0).use_empty();
        }
        if (isa<triton::AtomicCASOp>(op)) {
          // AtomicCAS has no slice implementation, even for producer-only IDs.
          hasUnsupportedAtomic = true;
        } else if (isa<triton::AtomicRMWOp>(op)) {
          auto taskIds = getAsyncTaskIds(op);
          hasUnsupportedAtomic |= taskIds.size() != 1 || taskIds.front() != 0;
        }
      });

      // The partitioner cannot represent live gathers or unsupported atomics.
      // Fall back before data partitioning can abort or omit an atomic effect.
      const char *unsupportedWork = nullptr;
      if (hasUnsupportedGather)
        unsupportedWork = "live gather in warp-specialized function";
      else if (hasUnsupportedAtomic)
        unsupportedWork = "unsupported atomic in warp-specialized function";

      if (unsupportedWork) {
        if (hasPreexistingTaskIds) {
          funcOp.emitError()
              << "warp specialization cannot fall back from " << unsupportedWork
              << " with preexisting async_task_id attributes";
          return signalPassFailure();
        }
        funcOp.walk([](Operation *op) { op->removeAttr("async_task_id"); });
        funcOp.walk([](scf::ForOp loop) {
          loop->removeAttr(triton::kWarpSpecializeAttrName);
        });
        return;
      }

      // Partition ops into parallel sub ops.
      if (doDataPartition(funcOp, numWarpGroups - 1)) {
        if (dumpIntermediateSteps) {
          ::mlir::triton::tools::mlirDumpsOrDbgs()
              << "// -----// WarpSpec internal IR Dump After: doDataPartition\n"
              << moduleOp << "\n\n\n";
        }
        success = true;
        break;
      }
      // Clear async_task.
    }
    if (!success) {
      mlir::emitError(
          getOperation()->getLoc(),
          "failed to partition the function into warp-specialized code");
      return signalPassFailure();
    }

    doCodePartition(funcOp, numStages);
    if (dumpIntermediateSteps) {
      ::mlir::triton::tools::mlirDumpsOrDbgs()
          << "// -----// WarpSpec internal IR Dump After: doCodePartition\n"
          << moduleOp << "\n\n\n";
    }
    doTokenLowering(funcOp, numWarpGroups - 1);
    invalidateWarpSpecializeBarriers(funcOp);
    // Clear num_stages to disable SWP.
    funcOp->walk([&](scf::ForOp forOp) {
      forOp->setAttr(mlir::triton::kNumStagesAttrName,
                     builder.getI32IntegerAttr(0));
    });
  }

  void runOnOperation() override {
    if (numStages <= 1)
      return;

    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
