#include "src/models/qwen38_flash_next/dense_kquant_lane.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/cpu_ops.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace lanes = gufo::models::qwen38_flash_next::kquant_lane;
using gufo::core::GgmlType;

/// The dense K-quant decode kernels' geometry, emulated on the CPU with the
/// kernels' own lane functions: every (block, lane) pair of the vector
/// kernel's K loop at each K split, and the WMMA kernel's chunk unpacking,
/// must reproduce the dequantize-then-F64 oracle on exactly representable
/// Q8_1 activation rows.
namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

struct Format {
  GgmlType type;
  std::size_t block;
  std::size_t bytes;
};

std::vector<std::uint8_t> RandomBlocks(const Format& f, std::size_t rows,
                                       std::size_t cols, std::uint32_t seed) {
  // Exact F16 scales of both signs and magnitudes, and positive minimums.
  constexpr std::uint16_t kScales[] = {0x1800, 0x1A40, 0x9500, 0x2100};
  constexpr std::uint16_t kMins[] = {0x1400, 0x1640, 0x1100, 0x0000};
  const std::size_t blocks = rows * cols / f.block;
  std::vector<std::uint8_t> bytes(blocks * f.bytes);
  for (auto& b : bytes)
    b = static_cast<std::uint8_t>(NextRandom(&seed));
  for (std::size_t i = 0; i < blocks; ++i) {
    std::uint8_t* block = bytes.data() + i * f.bytes;
    const std::uint16_t d = kScales[NextRandom(&seed) % 4];
    const std::uint16_t m = kMins[NextRandom(&seed) % 4];
    std::memcpy(block, &d, 2);
    std::memcpy(block + 2, &m, 2);
  }
  return bytes;
}

/// Q8_1 rows with power-of-two scales and |sum(q)| < 2048, so d and d*sum(q)
/// are exact in F16; `values` receives the dequantized activations.
std::vector<std::uint8_t> RandomQ8_1(std::size_t cols, std::uint32_t seed,
                                     std::vector<double>* values) {
  const std::size_t blocks = cols / 32;
  std::vector<std::uint8_t> bytes(blocks * lanes::kQ8_1Bytes);
  values->assign(cols, 0.0);
  for (std::size_t b = 0; b < blocks; ++b) {
    const int exponent = 5 + static_cast<int>(NextRandom(&seed) % 5);
    std::int8_t q[32];
    int sum = 0;
    for (int i = 0; i < 32; ++i) {
      q[i] = static_cast<std::int8_t>(
          static_cast<int>(NextRandom(&seed) % 255) - 127);
      sum += q[i];
    }
    Require(std::abs(sum) < 2048, "activation sum not exact in F16");
    // F16 2^-exponent, and d * sum(q) as an exact F16.
    const std::uint16_t d = static_cast<std::uint16_t>((15 - exponent) << 10);
    const double s = std::ldexp(static_cast<double>(sum), -exponent);
    std::uint16_t s_bits = 0;
    if (sum != 0) {
      int e = 0;
      const double frac = std::frexp(std::abs(s), &e);  // [0.5, 1)
      const int mant = static_cast<int>(std::ldexp(frac, 11)) - 1024;
      Require(e - 1 + 15 > 0, "activation sum below the F16 normal range");
      s_bits = static_cast<std::uint16_t>((sum < 0 ? 0x8000 : 0) |
                                          (e - 1 + 15) << 10 | mant);
    }
    std::uint8_t* block = bytes.data() + b * lanes::kQ8_1Bytes;
    std::memcpy(block, &d, 2);
    std::memcpy(block + 2, &s_bits, 2);
    std::memcpy(block + 4, q, 32);
    for (int i = 0; i < 32; ++i)
      (*values)[b * 32 + i] = std::ldexp(static_cast<double>(q[i]), -exponent);
  }
  return bytes;
}

template<class Lane>
double KernelRow(const std::uint8_t* row, const std::uint8_t* x,
                 std::size_t cols, int ksplit) {
  // The kernel's K loop: lane l of split part p starts at block
  // l / kLanes + p * per_iter and steps per_iter * ksplit, lane sum order.
  constexpr int per_iter = 32 / Lane::kLanes;
  const int blocks = static_cast<int>(cols / (Lane::kQ8Blocks * 32));
  double total = 0.0;
  std::vector<int> visits(static_cast<std::size_t>(blocks) * Lane::kLanes);
  for (int part = 0; part < ksplit; ++part) {
    for (int lane = 0; lane < 32; ++lane) {
      const int sub = lane % Lane::kLanes;
      float acc = 0.0F;
      for (int kb = lane / Lane::kLanes + part * per_iter; kb < blocks;
           kb += per_iter * ksplit) {
        ++visits[static_cast<std::size_t>(kb) * Lane::kLanes + sub];
        const Lane w(row + static_cast<std::size_t>(kb) * Lane::kBlockBytes,
                     sub);
        acc = w.Dot(x + static_cast<std::size_t>(kb) * Lane::kQ8Blocks *
                            lanes::kQ8_1Bytes,
                    sub, acc);
      }
      total += acc;
    }
  }
  for (const int v : visits)
    Require(v == 1, "a weight lane is visited " + std::to_string(v) + " times");
  return total;
}

/// The wide-batch WMMA kernel's data flow for one row and token: 64-wide
/// chunks, each two 16-byte quant parts unpacked into sub-blocks 2c (low
/// nibbles) and 2c + 1 (high), integer products with the matching Q8_1
/// blocks, then the sub-block scale and minimum.
template<bool five_bit>
double WmmaRow(const std::uint8_t* row, const std::uint8_t* x,
               std::size_t cols) {
  constexpr int block_bytes = five_bit ? 176 : 144;
  constexpr int quants = five_bit ? 48 : 16;
  double total = 0.0;
  for (std::size_t chunk = 0; chunk < cols / 64; ++chunk) {
    const int sb = static_cast<int>(chunk / 4);
    const int c = static_cast<int>(chunk % 4);
    const std::uint8_t* block =
        row + static_cast<std::size_t>(sb) * block_bytes;
    const auto scales = lanes::DecodeScales(lanes::Load128(block), c);
    std::uint32_t w[2][2][4];
    for (int part = 0; part < 2; ++part) {
      const auto q = lanes::Load128(block + quants + 32 * c + 16 * part);
      lanes::Words4 qh{};
      if constexpr (five_bit)
        qh = lanes::Load128(block + 16 + 16 * part);
      lanes::UnpackQuants<five_bit>(q, qh, c, w[0][part], w[1][part]);
    }
    for (int s = 0; s < 2; ++s) {
      const std::uint8_t* xb =
          x +
          (static_cast<std::size_t>(sb) * 8 + 2 * c + s) * lanes::kQ8_1Bytes;
      int dot = 0;
      for (int part = 0; part < 2; ++part)
        for (int i = 0; i < 4; ++i)
          dot = lanes::Dot4(w[s][part][i],
                            lanes::Load32(xb + 4 + 16 * part + 4 * i), dot);
      const auto head = lanes::Q8Head(xb);
      const double scale = s == 0 ? scales.d0 : scales.d1;
      const double minimum = s == 0 ? scales.m0 : scales.m1;
      total += scale * head.d * dot - minimum * head.s;
    }
  }
  return total;
}

template<class Lane>
void CheckFormat(const Format& f, std::size_t cols) {
  constexpr std::size_t kRows = 3;
  constexpr std::size_t kTokens = 3;
  const std::string tag =
      std::string(gufo::core::ToString(f.type)) + " k=" + std::to_string(cols);
  Require(static_cast<std::size_t>(Lane::kBlockBytes) == f.bytes,
          tag + " block size");
  const auto bytes =
      RandomBlocks(f, kRows, cols, 0x5EEDU ^ static_cast<std::uint32_t>(cols));
  qfn::TensorRef t;
  t.data = bytes.data();
  t.type = f.type;
  t.cols = cols;
  t.rows = kRows;
  std::vector<float> w(cols);
  const std::size_t row_bytes = cols / f.block * f.bytes;
  for (std::size_t n = 0; n < kTokens; ++n) {
    std::vector<double> xv;
    const auto x = RandomQ8_1(cols, 91U + static_cast<std::uint32_t>(n), &xv);
    for (std::size_t r = 0; r < kRows; ++r) {
      qfn::cpu::DequantizeRow(t, 0, r, w.data());
      double want = 0.0;
      double mag = 0.0;
      for (std::size_t i = 0; i < cols; ++i) {
        want += static_cast<double>(w[i]) * xv[i];
        mag += std::abs(static_cast<double>(w[i]) * xv[i]);
      }
      if constexpr (Lane::kQ8Blocks == 8) {
        const double got = WmmaRow<Lane::kBlockBytes == 176>(
            bytes.data() + r * row_bytes, x.data(), cols);
        Require(std::abs(got - want) <= 1e-5 * mag + 1e-7,
                tag + " row " + std::to_string(r) + " WMMA chunks: " +
                    std::to_string(got) + " vs oracle " + std::to_string(want));
      }
      for (const int ksplit : {1, 4}) {
        const double got = KernelRow<Lane>(bytes.data() + r * row_bytes,
                                           x.data(), cols, ksplit);
        Require(std::abs(got - want) <= 1e-5 * mag + 1e-7,
                tag + " row " + std::to_string(r) + " ksplit " +
                    std::to_string(ksplit) + ": " + std::to_string(got) +
                    " vs oracle " + std::to_string(want));
      }
    }
  }
}

}  // namespace

int main() {
  try {
    Require(lanes::HalfBits(0x3C00) == 1.0F &&
                lanes::HalfBits(0xC000) == -2.0F &&
                lanes::HalfBits(0x0001) == std::ldexp(1.0F, -24) &&
                lanes::HalfBits(0x7BFF) == 65504.0F,
            "F16 decoding");
    Require(lanes::Dot4(0x80FF017FU, 0x7F01FF7FU, 5) ==
                5 + 127 * 127 - 1 - 1 - 128 * 127,
            "signed byte dot product");
    const Format q4k{GgmlType::kQ4_K, 256, 144};
    const Format q5k{GgmlType::kQ5_K, 256, 176};
    const Format q51{GgmlType::kQ5_1, 32, 24};
    // Flash-Next reduction widths, and the shortest K-quant row.
    for (const std::size_t k : {256UL, 2560UL, 6144UL, 10240UL}) {
      CheckFormat<lanes::Q4KLane>(q4k, k);
      CheckFormat<lanes::Q5KLane>(q5k, k);
    }
    for (const std::size_t k : {32UL, 320UL, 640UL, 2560UL})
      CheckFormat<lanes::Q5_1Lane>(q51, k);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  std::cout << "dense K-quant lanes OK\n";
  return 0;
}
