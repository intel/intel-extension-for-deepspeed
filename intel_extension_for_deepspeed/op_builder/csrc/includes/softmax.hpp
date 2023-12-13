#pragma once

#include <stdio.h>
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#elif __has_include(<CL/sycl.hpp>)
#include <CL/sycl.hpp>
#else
#error "Unsupported compiler"
#endif
#include "custom_sycl_layers.hpp"

#include <fstream>

template <typename T>
class Softmax {
public:
    struct Config {
        size_t batchSize;
        size_t heads;
        size_t seq_length;
        size_t prob_depth;
        float temprature;
        bool mem_alloc;
        Config(size_t batch, size_t h, size_t seq, int prob_size = 0, bool mem_alloc = false)
            : batchSize(batch),
              heads(h),
              seq_length(seq),
              prob_depth(prob_size),
              temprature(1.0),
              mem_alloc(mem_alloc)
        {
        }
    };

    Softmax(Config config) : config_(config) {}

    ~Softmax() {}

    void Forward(int bsz, T* vals, const T* attn_mask, sycl::queue stream)
    {
        launch_attn_softmax<T>(vals, attn_mask, bsz, config_.heads, config_.seq_length, stream);
    }

    void Backward(int bsz, T* out_grad, const T* soft_out, sycl::queue stream)
    {
        launch_attn_softmax_backward_v2<T>(
            out_grad, soft_out, bsz, config_.heads, config_.seq_length, stream);
    }

    inline size_t GetProbDepth() const { return config_.prob_depth; }

    inline size_t GetBatchSize() const { return config_.batchSize; }

    inline size_t GetNumHeads() const { return config_.heads; }

    inline size_t GetSeqLength() const { return config_.seq_length; }

    inline void SetSeqLength(size_t seq_len) { config_.seq_length = seq_len; }

private:
    Config config_;
};
