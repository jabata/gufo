#pragma once
#include "common.hpp"

constexpr int MMVQ_MAX_BATCH_SIZE = 8;
constexpr int MMVQ_MAX_ROUTED_BATCH = 64;

// Resolve the first dense decode function before serving; launches no work.
int prepare_q8_decode();

// One expert's independent inputs, in groups of eight. The anchor also
// identifies an inactive slot that must be zeroed.
struct MoeBatchGroup {
  int expert;
  int anchor;
  int token[MMVQ_MAX_BATCH_SIZE];
  uint32_t slots[MMVQ_MAX_BATCH_SIZE];
};

// Tuned routed-vector batch limits for gfx1151.
constexpr __host__ __device__ int mmvq_moe_max_batch(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
                   type == GGML_TYPE_Q6_K
               ? 4
               : 8;
}

void mul_mat_vec_q8_dispatch(const void* weights, const void* gate,
                            const block_q8_1* input, float* output,
                            int k, int rows, int tokens, int input_stride,
                            hipStream_t stream);
// Dense Q4_K/Q5_K/Q6_K/Q5_1 matrix-vector product over 1..n Q8_1 input rows,
// reading the weights once per 64 rows. output is [tokens][rows]. Q4_K/Q5_K
// weights must be 16-byte aligned, Q5_1 weights 8-byte aligned. `route`
// (diagnostics) forces the vector (1) or WMMA (2) kernel.
void mul_mat_vec_dense_dispatch(const void* weights, ggml_type type,
                                const block_q8_1* input, float* output, int k,
                                int rows, int tokens, int input_stride,
                                hipStream_t stream, int route = 0);
void mul_mat_vec_moe_dispatch(const void* weights, ggml_type type,
                             const block_q8_1* input, const int32_t* ids, float* output,
                             int k, int rows, int tokens, int experts_used,
                             int input_stride, hipStream_t stream,
                             const int32_t* dst_slots = nullptr);
// Scratch ints mul_mat_vec_moe_gated needs for 2..MMVQ_MAX_BATCH_SIZE tokens:
// a group per slot and two anchor lists.
constexpr int mmvq_moe_group_ints(int tokens, int experts_used) {
  return tokens * experts_used * (tokens + 1) + 2 * (tokens * experts_used + 1);
}

constexpr size_t mmvq_moe_batch_group_bytes(int tokens, int experts_used) {
  return sizeof(int32_t) + size_t(tokens) * experts_used * sizeof(MoeBatchGroup);
}

// Q4_K above one token takes `groups` sized and zero-counted for
// group_moe_batch (mmvq_moe_batch_group_bytes). `route` (diagnostics) 1
// forces the scalar Q4_K kernels the lane kernel replaced.
void mul_mat_vec_moe_gated(const void* gate, const void* up, ggml_type type,
                           const block_q8_1* input, const int32_t* ids,
                           int32_t* groups, float* output, int k, int rows,
                           int tokens, int experts_used, int input_stride,
                           hipStream_t stream, int route = 0);

void mul_mat_vec_moe_batch_wave64(const void* gate, const void* up,
                                 const block_q8_1* input, const int32_t* groups,
                                 float* output, int k, int rows,
                                 int experts_used, int input_stride,
                                 hipStream_t stream);
