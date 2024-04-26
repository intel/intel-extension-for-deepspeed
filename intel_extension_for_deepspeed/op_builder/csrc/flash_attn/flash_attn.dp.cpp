#include <torch/extension.h>
#include "context.h"
#include "flash_attn.h"

// [Bs, Hn, Sl, Hs]
std::vector<torch::Tensor> flash_attn_fwd(torch::Tensor &q,
                                          torch::Tensor &k,
                                          torch::Tensor &v,
                                          uint32_t bs,
                                          uint32_t head_number,
                                          uint32_t seqlens,
                                          uint32_t head_size,
                                          float softmax_scale,
                                          float dropout_prob,
                                          uint64_t dropout_rand_seed,
                                          bool is_causal,
                                          bool is_training,
                                          bool is_dropout) {
    torch::Tensor q_ = q.transpose(1, 2).contiguous().transpose(1, 2);
    torch::Tensor k_ = k.transpose(1, 2).contiguous().transpose(1, 2);
    torch::Tensor v_ = v.transpose(1, 2).contiguous().transpose(1, 2);
    torch::Tensor output = torch::empty_like(q_);
    torch::Tensor softmax_L, dropout_mask;
    softmax_L = torch::empty({bs * head_number, 1, seqlens}, q.options()).to(at::kFloat);

    void *q_ptr = (void *)q_.data_ptr();
    void *k_ptr = (void *)k_.data_ptr();
    void *v_ptr = (void *)v_.data_ptr();
    void *output_ptr = (void *)output.data_ptr();
    void *softmax_L_ptr = (void *)softmax_L.data_ptr();
    void *drop_mask_ptr = nullptr;
    uint64_t dropout_rand_offset = 123;

    sycl::queue* stream = ::TrainingContext::Instance().GetCurrentStream();
    FlashAttention _flash_attn = FlashAttention();
    _flash_attn.Forward(
        *stream,
        output_ptr,
        softmax_L_ptr,
        bs,
        head_number,
        head_size,
        seqlens,
        seqlens,
        softmax_scale,
        q_ptr,
        k_ptr,
        v_ptr,
        drop_mask_ptr,
        dropout_prob,
        dropout_rand_seed,
        dropout_rand_offset,
        is_causal,
        is_training,
        is_dropout
    );
    return {output, softmax_L};
}

std::vector<torch::Tensor> flash_attn_bwd(torch::Tensor &gradout,
                                          torch::Tensor &q,
                                          torch::Tensor &k,
                                          torch::Tensor &v,
                                          torch::Tensor &out,
                                          uint32_t bs,
                                          uint32_t head_number,
                                          uint32_t seqlens,
                                          uint32_t head_size,
                                          float softmax_scale,
                                          float dropout_prob,
                                          uint64_t dropout_rand_seed,
                                          bool is_causal,
                                          bool is_dropout,
                                          torch::Tensor &softmax_L) {
    torch::Tensor q_ = q.transpose(1, 2).contiguous().transpose(1, 2);
    torch::Tensor k_ = k.transpose(1, 2).contiguous().transpose(1, 2);
    torch::Tensor v_ = v.transpose(1, 2).contiguous().transpose(1, 2);
    torch::Tensor out_ = out.transpose(1, 2).contiguous().transpose(1, 2);
    torch::Tensor grad_out_ = gradout.transpose(1, 2).contiguous().transpose(1, 2);

    torch::Tensor dq = torch::zeros_like(q_);
    torch::Tensor dk = torch::empty_like(k_);
    torch::Tensor dv = torch::empty_like(v_);
    torch::Tensor d_buffer = torch::empty_like(softmax_L);
    void *gradout_ptr = (void *)grad_out_.data_ptr();
    void *q_ptr = (void *)q_.data_ptr();
    void *k_ptr = (void *)k_.data_ptr();
    void *v_ptr = (void *)v_.data_ptr();
    void *out_ptr = (void *)out_.data_ptr();
    void *dq_ptr = (void *)dq.data_ptr();
    void *dk_ptr = (void *)dk.data_ptr();
    void *dv_ptr = (void *)dv.data_ptr();
    void *softmax_L_ptr = (void *)softmax_L.data_ptr();
    void *d_buffer_ptr = (void *)d_buffer.data_ptr();
    void *drop_mask_ptr = nullptr;
    uint64_t dropout_rand_offset = 123;

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
        d_buffer_ptr,
        bs,
        head_number,
        head_size,
        seqlens,
        seqlens,
        softmax_scale,
        q_ptr,
        k_ptr,
        v_ptr,
        drop_mask_ptr,
        dropout_prob,
        dropout_rand_seed,
        dropout_rand_offset,
        is_causal,
        is_dropout
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
