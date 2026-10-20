#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/cpu_ops.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
using gufo::core::GgmlType;

/// Dense Q4_K/Q5_K/Q6_K/Q5_1 projections (decode vector kernel and prefill
/// tiled GEMM) against the dequantize-then-F32 oracle, plus the decode
/// contract: a row's result does not depend on the batch width or on who
/// quantized its activations.
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

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

std::size_t BlockBytes(GgmlType type) {
  switch (type) {
    case GgmlType::kQ4_K:
      return 144;
    case GgmlType::kQ5_K:
      return 176;
    case GgmlType::kQ6_K:
      return 210;
    default:
      return 24;  // Q5_1
  }
}

std::size_t BlockSize(GgmlType type) {
  return type == GgmlType::kQ5_1 ? 32 : 256;
}

std::vector<std::uint8_t> RandomBlocks(GgmlType type, std::size_t rows,
                                       std::size_t cols, std::uint32_t seed) {
  const std::size_t blocks = rows * cols / BlockSize(type);
  std::vector<std::uint8_t> bytes(blocks * BlockBytes(type));
  for (auto& b : bytes)
    b = static_cast<std::uint8_t>(NextRandom(&seed));
  // F16 1/512 and 1/1024 for the super-block scale and minimum.
  const std::uint16_t d = 0x1800;
  const std::uint16_t m = 0x1400;
  for (std::size_t i = 0; i < blocks; ++i) {
    std::uint8_t* block = bytes.data() + i * BlockBytes(type);
    if (type == GgmlType::kQ6_K) {
      std::memcpy(block + 208, &d, 2);
    } else {
      std::memcpy(block, &d, 2);
      std::memcpy(block + 2, &m, 2);
    }
  }
  return bytes;
}

template<typename T>
T* Upload(const std::vector<T>& host, std::size_t extra_bytes = 0) {
  void* p = nullptr;
  const std::size_t bytes = host.size() * sizeof(T);
  CheckHip(hipMalloc(&p, bytes + extra_bytes), "allocation");
  CheckHip(hipMemset(p, 0, bytes + extra_bytes), "clear");
  CheckHip(hipMemcpy(p, host.data(), bytes, hipMemcpyHostToDevice), "upload");
  return static_cast<T*>(p);
}

std::vector<float> Download(const float* device, std::size_t count) {
  std::vector<float> host(count);
  CheckHip(hipMemcpy(host.data(), device, count * sizeof(float),
                     hipMemcpyDeviceToHost),
           "download");
  return host;
}

void CheckShape(GgmlType type, std::size_t m, std::size_t k) {
  const std::string tag = std::string(gufo::core::ToString(type)) + " " +
                          std::to_string(m) + "x" + std::to_string(k);
  const auto bytes = RandomBlocks(type, m, k, 0xC0FFEEU ^ (m * 31 + k));
  qfn::TensorRef t;
  t.data = bytes.data();
  t.type = type;
  t.cols = k;
  t.rows = m;
  constexpr std::size_t kMaxTokens = 72;
  std::uint32_t seed = 777U + static_cast<std::uint32_t>(k);
  std::vector<float> x(kMaxTokens * k);
  for (float& v : x)
    v = Uniform(&seed);
  // Oracle: dequantize, then F64 dot products, and sum |w x| for the bound.
  std::vector<double> want(kMaxTokens * m);
  std::vector<double> scale(kMaxTokens * m);
  std::vector<float> w(k);
  for (std::size_t r = 0; r < m; ++r) {
    qfn::cpu::DequantizeRow(t, 0, r, w.data());
    for (std::size_t n = 0; n < kMaxTokens; ++n) {
      double dot = 0.0;
      double mag = 0.0;
      for (std::size_t i = 0; i < k; ++i) {
        dot += static_cast<double>(w[i]) * x[n * k + i];
        mag += std::abs(static_cast<double>(w[i]) * x[n * k + i]);
      }
      want[n * m + r] = dot;
      scale[n * m + r] = mag;
    }
  }
  // The tail margin mirrors the device model's over-read allowance.
  auto* dw = Upload(bytes, 4096);
  auto* dx = Upload(x);
  float* dout = nullptr;
  CheckHip(hipMalloc(&dout, kMaxTokens * m * sizeof(float)), "output");
  void* dq8 = nullptr;
  CheckHip(hipMalloc(&dq8, qfn_mmq_q8_1_bytes(kMaxTokens, static_cast<int>(k))),
           "Q8_1 rows");
  const auto check = [&](const std::vector<float>& got, std::size_t n_tokens,
                         const char* path) {
    for (std::size_t i = 0; i < n_tokens * m; ++i) {
      Require(std::isfinite(got[i]) &&
                  std::abs(got[i] - want[i]) <= 1e-2 * scale[i] + 1e-6,
              tag + " " + path + " differs from the oracle at " +
                  std::to_string(i) + ": " + std::to_string(got[i]) + " vs " +
                  std::to_string(want[i]));
    }
  };
  const int type_id = static_cast<int>(type);
  std::vector<float> single;
  for (const std::size_t n : {1UL, 2UL, 4UL, 5UL, 8UL, 13UL, 64UL}) {
    Require(qfn_mmq_kquant_dense_vec(type_id, dw, dx, nullptr, dout,
                                     static_cast<int>(m), static_cast<int>(n),
                                     static_cast<int>(k), nullptr) == 0,
            tag + " vector launch");
    const auto got = Download(dout, n * m);
    check(got, n, "vector");
    if (n == 1) {
      single = got;
    } else {
      // Decode contract: the first row is bitwise the single-token result.
      Require(std::memcmp(got.data(), single.data(), m * sizeof(float)) == 0,
              tag + " vector result depends on the batch width");
    }
    // Pre-quantized rows (a producer's or Quantize's) give identical bits.
    Require(qfn_mmq_quantize_q8_1(dx, dq8, static_cast<int>(n),
                                  static_cast<int>(k), nullptr) == 0,
            "activation quantization");
    Require(qfn_mmq_kquant_dense_vec(type_id, dw, nullptr, dq8, dout,
                                     static_cast<int>(m), static_cast<int>(n),
                                     static_cast<int>(k), nullptr) == 0,
            tag + " pre-quantized vector launch");
    const auto preq = Download(dout, n * m);
    Require(std::memcmp(preq.data(), got.data(), n * m * sizeof(float)) == 0,
            tag + " pre-quantized rows change the result");
  }
  for (const std::size_t n : {1UL, 9UL, 64UL, 72UL}) {
    Require(qfn_mmq_kquant_dense(type_id, dw, dx, dout, static_cast<int>(m),
                                 static_cast<int>(n), static_cast<int>(k),
                                 nullptr) == 0,
            tag + " tiled launch");
    check(Download(dout, n * m), n, "tiled");
  }
  CheckHip(hipDeviceSynchronize(), "synchronize");
  (void)hipFree(dq8);
  (void)hipFree(dout);
  (void)hipFree(dx);
  (void)hipFree(dw);
}

}  // namespace

int main() {
  try {
    if (qfn_mmq_init(0) != 0)
      throw std::runtime_error("qfn_mmq_init failed");
    // Flash-Next dense geometries (reduced row counts): QKV/gate and shared
    // gate/up (k 2560), DeltaNet and attention output (k 6144), mixer down
    // (k 10240), and the Q5_1-only shared down (k 640) and mixer up (k 320).
    for (const auto type :
         {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K, GgmlType::kQ5_1}) {
      CheckShape(type, 640, 2560);
      CheckShape(type, 512, 6144);
      CheckShape(type, 320, 10240);
    }
    CheckShape(GgmlType::kQ5_1, 512, 640);
    CheckShape(GgmlType::kQ5_1, 1024, 320);
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
  std::cout << "dense K-quant projections OK\n";
  return 0;
}
