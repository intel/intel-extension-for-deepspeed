#include <torch/extension.h>
#include "context.h"
#include "flash_attn.hpp"

// [Bs, Hn, Sl, Hs]
std::vector<torch::Tensor> flash_attn_fwd(const torch::Tensor &q,
                                          const torch::Tensor &k,
                                          const torch::Tensor &v,
                                          const uint32_t bs,
                                          const uint32_t head_number,
                                          const uint32_t seqlens,
                                          const uint32_t head_size,
                                          const float softmax_scale,
                                          const float dropout_prob,
                                          const float dropout_scale,
                                          const uint64_t dropout_rand_seed,
                                          const bool causal,
                                          const bool return_softmax,
                                          const bool return_mask) {
    torch::Tensor output = torch::empty_like(q);
    torch::Tensor softmax_L, dropout_mask, softmax_m;
    if (return_softmax) {
        softmax_L = torch::empty({bs, head_number, seqlens, seqlens}, q.options()).to(at::kFloat);
    }
    else {
        softmax_L = torch::empty({bs * head_number, 1, seqlens}, q.options()).to(at::kFloat);
        softmax_m = torch::empty({bs * head_number, 1, seqlens}, q.options()).to(at::kFloat);
    }

    void *q_ptr = (void *)q.data_ptr();
    void *k_ptr = (void *)k.data_ptr();
    void *v_ptr = (void *)v.data_ptr();
    void *output_ptr = (void *)output.data_ptr();
    void *softmax_L_ptr = (void *)softmax_L.data_ptr();
    void *softmax_m_ptr = (void *)softmax_m.data_ptr();
    void *drop_mask_ptr = nullptr;
    if (return_mask) {
        dropout_mask = torch::empty({bs, head_number, seqlens, seqlens}, q.options()).to(at::kFloat);
        drop_mask_ptr = (void *)dropout_mask.data_ptr();
    }

    sycl::queue* stream = ::TrainingContext::Instance().GetCurrentStream();
    FlashAttention _flash_attn = FlashAttention();
    _flash_attn.Forward(
        *stream,
        output_ptr,
        softmax_L_ptr,
        softmax_m_ptr,
        bs,
        head_number,
        seqlens,
        head_size,
        softmax_scale,
        q_ptr,
        k_ptr,
        v_ptr,
        drop_mask_ptr,
        dropout_prob,
        dropout_scale,
        dropout_rand_seed,
        causal,
        return_softmax
    );
    return {output, softmax_L, softmax_m};
}

std::vector<torch::Tensor> flash_attn_bwd(const torch::Tensor &gradout,
                                          const torch::Tensor &q,
                                          const torch::Tensor &k,
                                          const torch::Tensor &v,
                                          const torch::Tensor &out,
                                          uint32_t bs,
                                          uint32_t head_number,
                                          uint32_t seqlens,
                                          uint32_t head_size,
                                          const float softmax_scale,
                                          const float dropout_prob,
                                          const float dropout_scale,
                                          const uint64_t dropout_rand_seed,
                                          const bool causal,
                                          const bool return_softmax,
                                          const torch::Tensor &softmax_L,
                                          const torch::Tensor &softmax_m) {
    torch::Tensor dq = torch::zeros_like(q);
    torch::Tensor dk = torch::empty_like(k);
    torch::Tensor dv = torch::empty_like(v);
    void *gradout_ptr = (void *)gradout.data_ptr();
    void *q_ptr = (void *)q.data_ptr();
    void *k_ptr = (void *)k.data_ptr();
    void *v_ptr = (void *)v.data_ptr();
    void *out_ptr = (void *)out.data_ptr();
    void *dq_ptr = (void *)dq.data_ptr();
    void *dk_ptr = (void *)dk.data_ptr();
    void *dv_ptr = (void *)dv.data_ptr();
    void *softmax_L_ptr = (void *)softmax_L.data_ptr();
    void *softmax_m_ptr = (void *)softmax_m.data_ptr();
    void *drop_mask_ptr = nullptr;
    void *grad_softmax = nullptr;

    sycl::queue* stream = ::TrainingContext::Instance().GetCurrentStream();
    FlashAttention _flash_attn = FlashAttention();
    _flash_attn.Backward(
        *stream,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        out_ptr,
        gradout_ptr,
        softmax_L_ptr,
        softmax_m_ptr,
        bs,
        head_number,
        seqlens,
        head_size,
        softmax_scale,
        q_ptr,
        k_ptr,
        v_ptr,
        drop_mask_ptr,
        dropout_prob,
        dropout_scale,
        dropout_rand_seed,
        causal,
        return_softmax
    );
    return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attn_fwd",
          &flash_attn_fwd,
          "Flash attention forward");
    m.def("flash_attn_bwd",
          &flash_attn_bwd,
          "Flash attention backward");
}
