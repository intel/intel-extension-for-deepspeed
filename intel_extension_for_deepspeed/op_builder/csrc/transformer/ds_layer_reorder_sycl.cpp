#include "common.hpp"
#include "context.hpp"
#include "custom_sycl_layers.hpp"
#include "general_kernels.hpp"

template <typename T>
std::vector<torch::Tensor> transform4d_0213(const torch::Tensor& input,
                                            int batch,
                                            int seq_len,
                                            int hidden_size,
                                            int num_heads,
                                            int trans_count)
{
    CHECK_INPUT(input);
    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    torch::Tensor output;
    if (trans_count == 3)
        // trans_count=3
        output = torch::empty({batch, seq_len, 3, num_heads, hidden_size / num_heads}, options);
    else
        // for 1 attn_o_inp, trans_count=1
        output = torch::empty({batch, seq_len, num_heads, hidden_size / num_heads}, options);

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    const T* input_ptr = (const T*)input.data_ptr();
    T* output_ptr = (T*)output.data_ptr();
    // trans_count=1
    // launch_transform4d_0213(output_ptr, input_ptr, batch, num_heads, seq_len,
    // hidden_size, q, 1);
    // trans_count=3
    launch_transform4d_0213(
        output_ptr, input_ptr, batch, num_heads, seq_len, hidden_size, q, trans_count);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> bias_add_transform_0213(const torch::Tensor& input,
                                                   const torch::Tensor& bias,
                                                   int batch,
                                                   int seq_len,
                                                   int hidden_size,
                                                   int num_heads)
{
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    auto output = torch::empty({3, batch, num_heads, seq_len, hidden_size / num_heads}, options);

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    const T* input_ptr = (const T*)input.data_ptr();
    const T* bias_ptr = (const T*)bias.data_ptr();
    T* output_ptr = (T*)output.data_ptr();
    launch_bias_add_transform_0213(
        output_ptr, input_ptr, bias_ptr, batch, seq_len, hidden_size, num_heads, q, 3);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> transform_0213(const torch::Tensor& input,
                                          int batch,
                                          int seq_len,
                                          int hidden_size,
                                          int num_heads)
{
    CHECK_INPUT(input);

    auto options = torch::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    auto output = torch::empty({batch, num_heads, seq_len, hidden_size / num_heads}, options);

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    const T* input_ptr = (const T*)input.data_ptr();
    T* output_ptr = (T*)output.data_ptr();

    launch_transform_0213(output_ptr, input_ptr, batch, seq_len, hidden_size, num_heads, q);
    return {output};
}

template <typename T>
std::vector<torch::Tensor> fused_add2(const torch::Tensor& input1,
                                      const torch::Tensor& input2,
                                      int batch,
                                      int seq_len,
                                      int hidden_size)
{
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);

    auto options = torch::TensorOptions()
                       .dtype(input1.options().dtype())
                       .layout(torch::kStrided)
                       .device(torch::kXPU)
                       .requires_grad(true);

    auto output = torch::empty({batch, seq_len, hidden_size}, options);

    sycl::queue q = ::SyclContext::Instance().GetCurrentStream();

    const T* input_ptr1 = (const T*)input1.data_ptr();
    const T* input_ptr2 = (const T*)input2.data_ptr();
    T* output_ptr = (T*)output.data_ptr();

    launch_fused_add2(output_ptr, input_ptr1, input_ptr2, batch, seq_len, hidden_size, q);
    return {output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_add2_fp32", &fused_add2<float>,
          "Fused add2 with fp32 (DPCPP)");
    m.def("fused_add2_fp16", &fused_add2<sycl::half>,
          "Fused add2 with fp16 (DPCPP)");
    m.def("fused_add2_bf16", &fused_add2<bf16>,
          "Fused add2 with bf16 (DPCPP)");
    m.def("transform_0213_fp32", &transform_0213<float>,
          "transform 0213 with fp32 (DPCPP)");
    m.def("transform_0213_fp16", &transform_0213<sycl::half>,
          "transform 0213 with fp16 (DPCPP)");
    m.def("transform_0213_bf16", &transform_0213<sycl::half>,
          "transform 0213 with bf16 (DPCPP)"); // simple memory copy
    m.def("bias_add_transform_0213_fp32", &bias_add_transform_0213<float>,
          "bias add transform 0213 with fp32 (DPCPP)");
    m.def("bias_add_transform_0213_fp16", &bias_add_transform_0213<sycl::half>,
          "bias add transform 0213 with fp16 (DPCPP)");
    m.def("bias_add_transform_0213_bf16", &bias_add_transform_0213<bf16>,
          "bias add transform 0213 with bf16 (DPCPP)");
    m.def("transform4d_0213_fp32", &transform4d_0213<float>,
          "transform4d 0213 with fp32 (DPCPP)");
    m.def("transform4d_0213_fp16", &transform4d_0213<sycl::half>,
          "transform4d 0213 with fp16 (DPCPP)");
    m.def("transform4d_0213_bf16", &transform4d_0213<sycl::half>,
          "transform4d 0213 with bf16 (DPCPP)"); // simple memory copy

}
