#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DENSE_KQUANT_LANE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DENSE_KQUANT_LANE_HPP_

#include <cstdint>

/// One lane's share of a dense K-quant decode dot product (Q4_K, Q5_K and
/// Q5_1 weights against Q8_1 activation rows). A lane decodes its weights
/// once into registers and reuses them for every token row of the batch; the
/// HIP vector kernel and the CPU emulation test both use these functions, so
/// the test checks the kernel's lane geometry against the dequantize oracle.
///
/// Geometry: a Q4_K/Q5_K super-block has eight lanes. Lane `sub` reads one
/// aligned 16-byte quant chunk (chunk sub / 2, half sub % 2), i.e. 16 weights
/// of sub-block 2c and 16 of sub-block 2c + 1, and the 16-byte block head
/// (scale, minimum and twelve 6-bit scale bytes) in one load. A Q5_1 block
/// has two lanes of 8 quant bytes each. The minimum term uses the Q8_1 block
/// sum (d * sum(q)) and is added by the half-0 lane only.
#if defined(__HIPCC__)
#define QFN_LANE __host__ __device__ __forceinline__
#else
#define QFN_LANE inline
#endif

namespace gufo::models::qwen38_flash_next::kquant_lane {

inline constexpr int kQ8_1Bytes = 36;

QFN_LANE std::uint32_t Load32(const std::uint8_t* p) {
  std::uint32_t v;
  __builtin_memcpy(&v, p, 4);
  return v;
}

struct Words4 {
  std::uint32_t w[4];
};

/// A 16-byte aligned load (one 128-bit load on the GPU).
QFN_LANE Words4 Load128(const std::uint8_t* p) {
  Words4 v;
  __builtin_memcpy(&v, __builtin_assume_aligned(p, 16), 16);
  return v;
}

struct Words2 {
  std::uint32_t w[2];
};

/// An 8-byte aligned load.
QFN_LANE Words2 Load64(const std::uint8_t* p) {
  Words2 v;
  __builtin_memcpy(&v, __builtin_assume_aligned(p, 8), 8);
  return v;
}

QFN_LANE float HalfBits(std::uint32_t h) {
#if defined(__HIP_DEVICE_COMPILE__)
  return static_cast<float>(
      __builtin_bit_cast(_Float16, static_cast<std::uint16_t>(h)));
#else
  const std::uint32_t sign = (h & 0x8000U) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1FU;
  std::uint32_t mant = h & 0x3FFU;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int e = -1;
      do {
        ++e;
        mant <<= 1;
      } while ((mant & 0x400U) == 0);
      bits = sign | static_cast<std::uint32_t>(112 - e) << 23 |
             (mant & 0x3FFU) << 13;
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000U | mant << 13;
  } else {
    bits = sign | (exp + 112) << 23 | mant << 13;
  }
  return __builtin_bit_cast(float, bits);
#endif
}

/// Signed 4 x int8 dot product with accumulate.
QFN_LANE int Dot4(std::uint32_t a, std::uint32_t b, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __builtin_amdgcn_sudot4(true, static_cast<int>(a), true,
                                 static_cast<int>(b), c, false);
#else
  for (int i = 0; i < 4; ++i)
    c += static_cast<std::int8_t>(a >> (8 * i)) *
         static_cast<std::int8_t>(b >> (8 * i));
  return c;
#endif
}

// Correctly rounded products and sums: fast-math must not reassociate the
// lane's chain, so a token's result is the same at every batch width.
QFN_LANE float Mul(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}
QFN_LANE float Fma(float a, float b, float c) {
#if defined(__HIP_DEVICE_COMPILE__)
  return __fmaf_rn(a, b, c);
#else
  return __builtin_fmaf(a, b, c);
#endif
}

/// Spreads four bits into bit 4 of each byte.
QFN_LANE std::uint32_t HighBits(std::uint32_t nibble) {
  return ((nibble & 15U) * 0x00204081U & 0x01010101U) << 4;
}

/// A Q8_1 block's scale and its d * sum(q).
struct Q8Scale {
  float d;
  float s;
};
QFN_LANE Q8Scale Q8Head(const std::uint8_t* block) {
  const std::uint32_t ds = Load32(block);
  return {HalfBits(ds & 0xFFFFU), HalfBits(ds >> 16)};
}

/// The scales and minimums (times the super-block's d and dmin) of
/// sub-blocks 2c and 2c + 1 from a Q4_K/Q5_K block head (ggml
/// get_scale_min_k4).
struct SubBlockScales {
  float d0, d1, m0, m1;
};
QFN_LANE SubBlockScales DecodeScales(const Words4& head, int c) {
  const float d = HalfBits(head.w[0] & 0xFFFFU);
  const float dmin = HalfBits(head.w[0] >> 16);
  const int shift = 16 * (c & 1);
  const std::uint32_t a = (head.w[1] >> shift) & 0xFFFFU;  // scales[0..3]
  const std::uint32_t b = (head.w[2] >> shift) & 0xFFFFU;  // scales[4..7]
  const std::uint32_t e = (head.w[3] >> shift) & 0xFFFFU;  // scales[8..11]
  std::uint32_t sc;
  std::uint32_t mn;
  if (c < 2) {
    sc = a & 0x3F3FU;
    mn = b & 0x3F3FU;
  } else {
    sc = (e & 0x0F0FU) | ((a & 0xC0C0U) >> 2);
    mn = ((e >> 4) & 0x0F0FU) | ((b & 0xC0C0U) >> 2);
  }
  return {Mul(d, static_cast<float>(sc & 0xFFU)),
          Mul(d, static_cast<float>(sc >> 8)),
          Mul(dmin, static_cast<float>(mn & 0xFFU)),
          Mul(dmin, static_cast<float>(mn >> 8))};
}

/// Sixteen quant bytes of chunk c as int8 weights: the low nibbles are
/// sub-block 2c, the high nibbles 2c + 1; Q5_K adds bits 2c and 2c + 1 of
/// the matching high-bit bytes.
template<bool five_bit>
QFN_LANE void UnpackQuants(const Words4& q, const Words4& qh, int c,
                           std::uint32_t* lo, std::uint32_t* hi) {
  for (int i = 0; i < 4; ++i) {
    lo[i] = q.w[i] & 0x0F0F0F0FU;
    hi[i] = (q.w[i] >> 4) & 0x0F0F0F0FU;
    if constexpr (five_bit) {
      lo[i] |= ((qh.w[i] >> (2 * c)) & 0x01010101U) << 4;
      hi[i] |= ((qh.w[i] >> (2 * c + 1)) & 0x01010101U) << 4;
    }
  }
}

/// Q4_K and Q5_K (same head and 6-bit scale packing; Q5_K adds 32 bytes of
/// high bits before the quants).
template<bool five_bit>
struct KQuantLane {
  static constexpr int kLanes = 8;
  static constexpr int kBlockBytes = five_bit ? 176 : 144;
  static constexpr int kQ8Blocks = 8;
  std::uint32_t lo[4];
  std::uint32_t hi[4];
  float d0, d1, m0, m1;

  KQuantLane() = default;
  QFN_LANE KQuantLane(const std::uint8_t* block, int sub) {
    const int c = sub / 2;
    const int h = sub % 2;
    const SubBlockScales s = DecodeScales(Load128(block), c);
    d0 = s.d0;
    d1 = s.d1;
    m0 = h == 0 ? s.m0 : 0.0F;
    m1 = h == 0 ? s.m1 : 0.0F;
    const int quants = five_bit ? 48 : 16;
    const Words4 q = Load128(block + quants + 32 * c + 16 * h);
    Words4 qh{};
    if constexpr (five_bit)
      qh = Load128(block + 16 + 16 * h);
    UnpackQuants<five_bit>(q, qh, c, lo, hi);
  }

  /// `x` is the first of the super-block's eight Q8_1 blocks.
  QFN_LANE float Dot(const std::uint8_t* x, int sub, float acc) const {
    const int c = sub / 2;
    const int h = sub % 2;
    const std::uint8_t* xa = x + (2 * c) * kQ8_1Bytes;
    const std::uint8_t* xb = xa + kQ8_1Bytes;
    int dot_a = 0;
    int dot_b = 0;
    for (int i = 0; i < 4; ++i) {
      dot_a = Dot4(lo[i], Load32(xa + 4 + 16 * h + 4 * i), dot_a);
      dot_b = Dot4(hi[i], Load32(xb + 4 + 16 * h + 4 * i), dot_b);
    }
    const Q8Scale sa = Q8Head(xa);
    const Q8Scale sb = Q8Head(xb);
    acc = Fma(Mul(d0, sa.d), static_cast<float>(dot_a), acc);
    acc = Fma(Mul(d1, sb.d), static_cast<float>(dot_b), acc);
    acc = Fma(-m0, sa.s, acc);
    return Fma(-m1, sb.s, acc);
  }
};

using Q4KLane = KQuantLane<false>;
using Q5KLane = KQuantLane<true>;

/// Q5_1: d, m, 32 high bits, 16 quant bytes; value = d * q + m.
struct Q5_1Lane {
  static constexpr int kLanes = 2;
  static constexpr int kBlockBytes = 24;
  static constexpr int kQ8Blocks = 1;
  std::uint32_t lo[2];
  std::uint32_t hi[2];
  float d, m;

  Q5_1Lane() = default;
  QFN_LANE Q5_1Lane(const std::uint8_t* block, int sub) {
    const Words2 head = Load64(block);
    const Words2 q = Load64(block + 8 + 8 * sub);
    d = HalfBits(head.w[0] & 0xFFFFU);
    m = sub == 0 ? HalfBits(head.w[0] >> 16) : 0.0F;
    const std::uint32_t qh = head.w[1];
    for (int i = 0; i < 2; ++i) {
      const int first = 8 * sub + 4 * i;
      lo[i] = (q.w[i] & 0x0F0F0F0FU) | HighBits(qh >> first);
      hi[i] = ((q.w[i] >> 4) & 0x0F0F0F0FU) | HighBits(qh >> (16 + first));
    }
  }

  QFN_LANE float Dot(const std::uint8_t* x, int sub, float acc) const {
    int dot = 0;
    for (int i = 0; i < 2; ++i) {
      dot = Dot4(lo[i], Load32(x + 4 + 8 * sub + 4 * i), dot);
      dot = Dot4(hi[i], Load32(x + 20 + 8 * sub + 4 * i), dot);
    }
    const Q8Scale s = Q8Head(x);
    acc = Fma(Mul(d, s.d), static_cast<float>(dot), acc);
    return Fma(m, s.s, acc);
  }
};

}  // namespace gufo::models::qwen38_flash_next::kquant_lane

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DENSE_KQUANT_LANE_HPP_
