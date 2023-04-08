#include <torch/extension.h>
#include <stdexcept>
#include <vector>
#include "compatible.h"
#include "context.hpp"
#include "inference_sycl_layers.h"


enum class TransformerType : uint8_t { UNKNOWN = 0, GPTType = 1, BERTType = 2 };

// NOTE: this is a temporary and dodgy solution to distinguish GPT and BERT style models
// based on the dimensions of the corresponding attention mask.
inline auto infer_transformer_type(at::Tensor& attn_mask) -> TransformerType
{
    auto attn_mask_num_dims = attn_mask.sizes().size();

    if (attn_mask_num_dims > 2) {
        return TransformerType::GPTType;
    } else if (attn_mask_num_dims == 2) {
        return TransformerType::BERTType;
    } else {
        return TransformerType::UNKNOWN;
    }
}


// infer stride of attention mask memory layout based on the model type.
inline auto get_attn_mask_stride(at::Tensor& attn_mask) -> int
{
    auto trnsfrmr_type = infer_transformer_type(attn_mask);

    if (trnsfrmr_type == TransformerType::GPTType) {
        return attn_mask.size(2);
    } else if (trnsfrmr_type == TransformerType::BERTType) {
        // Bert style models have always a mask stride of 1.
        return 1;
    } else if (trnsfrmr_type == TransformerType::UNKNOWN) {
        return 0;
    }

    // this is just to make the compiler happy.
    return 0;
}

template <typename T>
at::Tensor ds_softmax(at::Tensor& attn_scores,
                      at::Tensor& attn_mask,
                      at::Tensor& alibi,
                      bool triangular,
                      bool recompute,
                      bool local_attention,
                      int window_size,
                      bool async_op,
                      float layer_scale,
                      int head_offset,
                      int mp_size)
{
    auto attn_scores_c = attn_scores.contiguous();
    int bsz = attn_scores_c.size(0);

    int seq_len = attn_scores_c.size(1);
    int len = attn_scores_c.sizes().size();
    if (len > 2) seq_len = attn_scores_c.size(2);

    int soft_len = attn_scores_c.size(2);
    if (len > 3) soft_len = attn_scores_c.size(3);

    int heads = 1;
    if (len > 1) heads = attn_scores_c.size(1);

    auto mask_stride = get_attn_mask_stride(attn_mask);

    launch_attn_softmax_v2((T*)attn_scores_c.data_ptr(),
                           (attn_mask.sizes().size() > 1 ? (T*)attn_mask.data_ptr() : nullptr),
                           (alibi.sizes().size() > 1 ? (T*)alibi.data_ptr() : nullptr),
                           layer_scale,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           heads,
                           seq_len,
                           soft_len,
                           head_offset,
                           mask_stride,
                           mp_size,
                           SyclContext::Instance().GetCurrentStream());

    return attn_scores_c;
}


template <typename T>
at::Tensor& residual_add_bias(at::Tensor& hidden_state,
                              at::Tensor& residual,
                              const at::Tensor& attention_output,
                              const at::Tensor& attention_bias,
                              const at::Tensor& final_bias,
                              const int mp_size,
                              const bool mlp_after_attn,
                              const bool add_bias,
                              const bool preln)
{
    int bsz = residual.size(0) * residual.size(1);
    int hidden_size = residual.size(2);
    if (mlp_after_attn)
        launch_bias_residual(static_cast<T*>(residual.data_ptr()),
                             static_cast<T*>(hidden_state.data_ptr()),
                             static_cast<T*>(attention_output.data_ptr()),
                             static_cast<T*>(final_bias.data_ptr()),
                             static_cast<T*>(attention_bias.data_ptr()),
                             bsz,
                             hidden_size,
                             mp_size,
                             preln,
                             SyclContext::Instance().GetCurrentStream());
    /* else */
    /*     launch_gptj_residual_add<T>( */
    /*         static_cast<T*>(residual.data_ptr()), */
    /*         static_cast<T*>(hidden_state.data_ptr()), */
    /*         static_cast<T*>(attention_output.data_ptr()), */
    /*         static_cast<T*>(final_bias.data_ptr()), */
    /*         static_cast<T*>((add_bias ? attention_bias.data_ptr() : nullptr)), */
    /*         hidden_size, */
    /*         bsz, */
    /*         mp_size, */
    /*         Context::Instance().GetCurrentStream()); */
    return residual;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax_fp32", &ds_softmax<float>, "DeepSpeed SoftMax with fp32 (SYCL)");
    m.def("softmax_bf16", &ds_softmax<bf16>, "DeepSpeed SoftMax with bf16 (SYCL)");
    m.def("softmax_fp16", &ds_softmax<half>, "DeepSpeed SoftMax with fp16 (SYCL)");
    m.def("residual_add_bias_fp32",
          &residual_add_bias<float>,
          "DeepSpeed residual add with fp32 (CUDA)");
    m.def("residual_add_bias_bf16",
          &residual_add_bias<bf16>,
          "DeepSpeed residual add with bf16 (CUDA)");
    m.def("residual_add_bias_fp16",
          &residual_add_bias<half>,
          "DeepSpeed residual add with fp16 (CUDA)");
}