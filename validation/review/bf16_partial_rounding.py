"""Demonstrate why rounding each K partial differs from rounding their sum."""

import struct


def as_bf16_round_to_nearest_even(value: float) -> float:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack(">f", struct.pack(">I", rounded & 0xFFFF0000))[0]


def main() -> None:
    first = 1.0 + 1.0 / 256.0
    second = -1.0
    once = as_bf16_round_to_nearest_even(first + second)
    partials = (
        as_bf16_round_to_nearest_even(first)
        + as_bf16_round_to_nearest_even(second)
    )
    print(f"partials: {first}, {second}")
    print(f"round once after sum: {once}")
    print(f"round each partial before sum: {partials}")
    assert once == 1.0 / 256.0
    assert partials == 0.0


if __name__ == "__main__":
    main()
