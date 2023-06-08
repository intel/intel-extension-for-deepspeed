#include <torch/extension.h>
#include "context.hpp"
#include "flash_attn.hpp"

// [Bs, Hn, Sl, Hs]
std::vector<torch::Tensor> flash_atten_fwd(const torch::Tensor &q,
                                           const torch::Tensor &k,
                                           const torch::Tensor &v,
                                           uint32_t &bs,
                                           uint32_t &head_number,
                                           uint32_t &seqlens,
                                           uint32_t &head_size,
                                           const float dropout_p,
                                           const float softmax_scale,
                                           const bool causal,
                                           const bool return_attn_probs) {
    float dropout_scale = 1 / (1 - dropout_p);
    torch::Tensor output = torch::empty({bs, seqlens, head_number, head_size}, q.options());
    torch::Tensor out_buffer = torch::empty({bs, seqlens, head_number, head_size}, q.options());
    torch::Tensor softmax_res;
    if (return_attn_probs) {
        softmax_res = torch::empty({bs, head_number, seqlens, seqlens}, q.options());
    }

    void *q_ptr = (void *)q.data_ptr();
    void *k_ptr = (void *)k.data_ptr();
    void *v_ptr = (void *)v.data_ptr();
    void *output_ptr = (void *)output.data_ptr();
    void *out_buffer_ptr = (void *)out_buffer.data_ptr();
    void *softmax_res_ptr = (void *)softmax_res.data_ptr();

    sycl::queue* stream = ::SyclContext::Instance().GetCurrentStream();
    FlashAttention _flash_atten = FlashAttention();
    _flash_atten.Forward(
        stream,
        output_ptr,
        out_buffer_ptr,
        softmax_res_ptr,
        bs,
        head_number,
        seqlens,
        head_size,
        softmax_scale,
        q_ptr,
        k_ptr,
        v_ptr,
        dropout_scale=dropout_scale,
        store_softmax_out=return_attn_probs,
        is_causal=causal
    );
    return {output, softmax_res};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_atten_fwd",
          &flash_atten_fwd,
          "Flash attention forward for fp32");
}
