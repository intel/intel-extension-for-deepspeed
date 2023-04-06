#include <torch/extension.h>
#include <stdexcept>
#include <vector>
#include "compatible.h"
#include "context.hpp"
#include "inference_sycl_layers.h"
#include "onednn_wrappers.hpp"


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
void allocate_workspace(unsigned hidden_dim,
                        unsigned num_heads,
                        unsigned prompt_length,
                        unsigned batch_size,
                        unsigned num_layers,
                        unsigned mp_size = 1,
                        bool external_cache = false,
                        unsigned rank = 0,
                        unsigned max_out_tokens = 1024)
{
    SyclContext::Instance().GenWorkSpace(num_layers,
                                         num_heads,
                                         batch_size,
                                         prompt_length,
                                         hidden_dim,
                                         mp_size,
                                         external_cache,
                                         sizeof(T),
                                         rank,
                                         max_out_tokens);
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

template <typename T>
void ds_layer_norm_internal(T* workspace,
                            at::Tensor& input,
                            at::Tensor& gamma,
                            at::Tensor& beta,
                            float epsilon)
{
    int bsz = input.size(0) * input.size(1);
    launch_fused_ln(workspace,
                    (const T*)input.data_ptr(),
                    (const T*)gamma.data_ptr(),
                    (const T*)beta.data_ptr(),
                    epsilon,
                    bsz,
                    input.size(2),
                    SyclContext::Instance().GetCurrentStream());
}

template <typename T>
at::Tensor qkv_unfused_sycl(at::Tensor& output,
                              at::Tensor& input,
                              at::Tensor& weight,
                              at::Tensor& q_scale,
                              at::Tensor& bias,
                              at::Tensor& gamma,
                              at::Tensor& beta,
                              const float epsilon,
                              bool add_bias,
                              bool q_int8)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)SyclContext::Instance().GetWorkSpace();
    workspace += (3 * bsz * input.size(2));
    ds_layer_norm_internal<T>(workspace, input, gamma, beta, epsilon);

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;

    /* cublasSetStream(Context::Instance().GetCublasHandle(), */
    /*                 Context::Instance().GetCurrentStream()); */
    onednn_matmul_ex(SyclContext::Instance().GetCurrentStream(),
                     false,
                     false,
                     bsz,
                     weight.size(1),
                     input.size(2),
                     alpha,
                     gemm_beta,
                     workspace,
                     (T*)weight.data_ptr(),
                     (T*)output.data_ptr());
    
    if (add_bias)
        launch_bias_add((T*)output.data_ptr(),
                        (T*)bias.data_ptr(),
                        q_int8 ? weight.size(0) : weight.size(1),
                        bsz,
                        SyclContext::Instance().GetCurrentStream());

    auto output_stride = c10::TensorType::contiguousStridesOf(input.sizes());
    return at::from_blob(workspace, 
                            input.sizes(), 
                            output_stride,
                            nullptr,
                            input.options(),
                            input.device());
}

template <typename T>
std::vector<at::Tensor> ds_qkv_gemm(at::Tensor& input,
                                    at::Tensor& weight,
                                    at::Tensor& q_scale,
                                    at::Tensor& bias,
                                    at::Tensor& gamma,
                                    at::Tensor& beta,
                                    const float epsilon,
                                    bool add_bias,
                                    unsigned num_layers,
                                    bool external_cache,
                                    unsigned mp_size,
                                    unsigned rank,
                                    bool q_int8)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)SyclContext::Instance().GetWorkSpace();
    int out_size = q_int8 ? weight.size(0) : weight.size(1);

    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(false);

    auto output_stride = c10::TensorType::contiguousStridesOf({input.size(0), input.size(1), out_size});
    auto output = at::from_blob(workspace, 
                                {input.size(0), input.size(1), out_size}, 
                                output_stride, 
                                nullptr, 
                                options,
                                input.device());
    auto inp_norm = qkv_unfused_sycl<T>(
        output, input, weight, q_scale, bias, gamma, beta, epsilon, add_bias, q_int8);

    return {output, inp_norm};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax_fp32", &ds_softmax<float>, "DeepSpeed SoftMax with fp32 (SYCL)");
    m.def("softmax_bf16", &ds_softmax<bf16>, "DeepSpeed SoftMax with bf16 (SYCL)");
    m.def("softmax_fp16", &ds_softmax<half>, "DeepSpeed SoftMax with fp16 (SYCL)");
    m.def("residual_add_bias_fp32",
          &residual_add_bias<float>,
          "DeepSpeed residual add with fp32 (SYCL)");
    m.def("residual_add_bias_bf16",
          &residual_add_bias<bf16>,
          "DeepSpeed residual add with bf16 (SYCL)");
    m.def("residual_add_bias_fp16",
          &residual_add_bias<half>,
          "DeepSpeed residual add with fp16 (SYCL)");
    m.def("qkv_gemm_bf16", &ds_qkv_gemm<bf16>, "DeepSpeed qkv gemm with bf16 (SYCL)");
    m.def("allocate_workspace_fp32",
          &allocate_workspace<float>,
          "DeepSpeed memory allocation for GPT inference with fp32 (SYCL)");
    m.def("allocate_workspace_bf16",
          &allocate_workspace<bf16>,
          "DeepSpeed memory allocation for GPT inference with bf16 (SYCL)");
    m.def("allocate_workspace_fp16",
          &allocate_workspace<sycl::half>,
          "DeepSpeed memory allocation for GPT inference with fp16 (SYCL)");
}
