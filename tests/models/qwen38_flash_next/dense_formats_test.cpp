#include "src/models/qwen38_flash_next/dense_formats.hpp"

#include <algorithm>
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
using gufo::core::GgmlType;

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

float Uniform(std::uint32_t* state) {
  return static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

std::uint16_t Half(float value) {
  // Exact for the small power-of-two-scaled values used here.
  const int exponent = static_cast<int>(std::floor(std::log2(value)));
  const float mantissa = value / std::ldexp(1.0F, exponent) - 1.0F;
  return static_cast<std::uint16_t>(((exponent + 15) << 10) |
                                    static_cast<int>(mantissa * 1024.0F));
}

std::size_t BlockBytes(GgmlType type) {
  switch (type) {
    case GgmlType::kQ4_K:
      return 144;
    case GgmlType::kQ5_K:
      return 176;
    case GgmlType::kQ6_K:
      return 210;
    case GgmlType::kQ5_1:
      return 24;
    default:
      throw std::logic_error("unexpected format");
  }
}

std::size_t BlockSize(GgmlType type) {
  return type == GgmlType::kQ5_1 ? 32 : 256;
}

/// Random but well-formed blocks: random codes and sub-block scales, with
/// positive F16 super-block scales so every row has a realistic magnitude.
std::vector<std::uint8_t> RandomBlocks(GgmlType type, std::size_t rows,
                                       std::size_t cols, std::uint32_t seed) {
  const std::size_t blocks = rows * cols / BlockSize(type);
  std::vector<std::uint8_t> bytes(blocks * BlockBytes(type));
  for (auto& b : bytes)
    b = static_cast<std::uint8_t>(NextRandom(&seed));
  for (std::size_t i = 0; i < blocks; ++i) {
    std::uint8_t* block = bytes.data() + i * BlockBytes(type);
    const std::uint16_t d = Half(1.0F / 512.0F);
    const std::uint16_t m = Half(1.0F / 1024.0F);
    switch (type) {
      case GgmlType::kQ4_K:
      case GgmlType::kQ5_K:
      case GgmlType::kQ5_1:
        std::memcpy(block, &d, 2);
        std::memcpy(block + 2, &m, 2);
        break;
      case GgmlType::kQ6_K:
        std::memcpy(block + 208, &d, 2);
        break;
      default:
        break;
    }
  }
  return bytes;
}

/// The vector kernels' activation rounding: x quantized per 32-element block
/// to int8 with scale amax / 127 (llama.cpp's block_q8_1).
std::vector<float> RoundQ8_1(const std::vector<float>& x) {
  std::vector<float> out(x.size());
  for (std::size_t b = 0; b < x.size(); b += 32) {
    float amax = 0.0F;
    for (std::size_t i = b; i < b + 32; ++i)
      amax = std::max(amax, std::abs(x[i]));
    const float d = amax / 127.0F;
    for (std::size_t i = b; i < b + 32; ++i)
      out[i] = d == 0.0F ? 0.0F : std::round(x[i] / d) * d;
  }
  return out;
}

void CheckFormatContract() {
  for (const auto type :
       {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K, GgmlType::kQ5_1}) {
    Require(qfn::IsDenseKQuant(type), "sub-8-bit dense format rejected");
    Require(qfn::IsStackableDense(type), "same-format stack rejected");
    Require(
        std::find(qfn::kDenseMatrixTypes.begin(), qfn::kDenseMatrixTypes.end(),
                  type) != qfn::kDenseMatrixTypes.end(),
        "binder does not accept a dense K-quant format");
  }
  for (const auto type :
       {GgmlType::kQ8_0, GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32}) {
    Require(!qfn::IsDenseKQuant(type), "existing format rerouted");
    Require(
        std::find(qfn::kDenseMatrixTypes.begin(), qfn::kDenseMatrixTypes.end(),
                  type) != qfn::kDenseMatrixTypes.end(),
        "existing dense format dropped");
  }
  Require(qfn::IsStackableDense(GgmlType::kQ8_0), "Q8_0 stack dropped");
  Require(!qfn::IsStackableDense(GgmlType::kBF16), "BF16 stack accepted");
  Require(!qfn::IsDenseKQuant(GgmlType::kIQ4_NL), "IQ4_NL dense accepted");
  Require(
      qfn::SelectKQuantDenseKernel(false) == qfn::KQuantDenseKernel::kVector,
      "decode does not take the vector kernel");
  Require(qfn::SelectKQuantDenseKernel(true) == qfn::KQuantDenseKernel::kTiled,
          "prefill does not take the tiled kernel");
}

/// The reduction widths of the Flash-Next dense projections: hidden 2560,
/// DeltaNet value 6144, attention 6144, MTP fc 5120, mixer down 10240,
/// shared-expert down 640, mixer up 320. Superblock formats need k % 256.
void CheckShapes() {
  struct Shape {
    std::uint64_t cols;
    bool superblock;
  };
  for (const Shape s :
       {Shape{2560, true}, Shape{6144, true}, Shape{5120, true},
        Shape{10240, true}, Shape{640, false}, Shape{320, false}}) {
    for (const auto type :
         {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K, GgmlType::kQ5_1}) {
      qfn::TensorRef t;
      t.type = type;
      t.cols = s.cols;
      const bool legal = type == GgmlType::kQ5_1 || s.superblock;
      const std::size_t expected =
          legal ? s.cols / BlockSize(type) * BlockBytes(type) : 0;
      Require(t.RowBytes() == expected,
              "row geometry for k=" + std::to_string(s.cols));
    }
  }
  // The MTP fc projection splits each row in half on a block boundary.
  for (const auto type : {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K}) {
    qfn::TensorRef whole;
    whole.type = type;
    whole.cols = 5120;
    qfn::TensorRef half = whole;
    half.cols = 2560;
    Require(whole.RowBytes() == 2 * half.RowBytes(),
            "MTP fc split leaves a partial block");
  }
}

/// The dense oracle (dequantize, then F32/F64 math) against the vector
/// kernels' rounding model. The bound below is what the GPU operator test
/// allows its K-quant kernels relative to the same oracle.
void CheckOracle() {
  constexpr std::size_t kRows = 24;
  for (const auto type :
       {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K, GgmlType::kQ5_1}) {
    for (const std::uint64_t cols : {2560ULL, 6144ULL, 640ULL, 320ULL}) {
      if (type != GgmlType::kQ5_1 && cols % 256 != 0)
        continue;
      const auto bytes = RandomBlocks(type, kRows, cols, 0x9E3779B9U ^ cols);
      qfn::TensorRef t;
      t.data = bytes.data();
      t.type = type;
      t.cols = cols;
      t.rows = kRows;
      Require(t.SizeBytes() == bytes.size(), "tensor size");
      std::uint32_t seed = 12345U + static_cast<std::uint32_t>(cols);
      std::vector<float> x(cols);
      for (float& v : x)
        v = Uniform(&seed);
      std::vector<float> y(kRows);
      qfn::cpu::MatVec(t, 0, x, y);
      const auto xq = RoundQ8_1(x);
      std::vector<float> w(cols);
      for (std::size_t r = 0; r < kRows; ++r) {
        qfn::cpu::DequantizeRow(t, 0, r, w.data());
        double dot = 0.0;
        double quantized = 0.0;
        double magnitude = 0.0;
        for (std::size_t i = 0; i < cols; ++i) {
          dot += static_cast<double>(w[i]) * x[i];
          quantized += static_cast<double>(w[i]) * xq[i];
          magnitude += std::abs(static_cast<double>(w[i]) * x[i]);
        }
        Require(magnitude > 0.0, "degenerate random row");
        Require(std::abs(dot - y[r]) <= 1e-5 * magnitude,
                "MatVec disagrees with its own dequantization");
        // int8 activations: per-element error <= d/2 = amax/254, so the
        // dot moves by well under 1% of sum |w x| for these rows.
        Require(std::abs(quantized - dot) <= 1e-2 * magnitude,
                std::string("Q8_1 rounding exceeds the bound for ") +
                    std::string(gufo::core::ToString(type)) +
                    " k=" + std::to_string(cols));
      }
    }
  }
}

}  // namespace

int main() {
  try {
    CheckFormatContract();
    CheckShapes();
    CheckOracle();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  std::cout << "dense formats OK\n";
  return 0;
}
