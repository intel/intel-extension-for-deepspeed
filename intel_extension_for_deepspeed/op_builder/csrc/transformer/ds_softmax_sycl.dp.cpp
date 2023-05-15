#include "common.hpp"
#include "context.hpp"
#include "softmax.hpp"

template <typename T>
std::vector<torch::Tensor> softmax_forward(int bsz,
                                           int seq_len,
                                           int num_heads,
                                           torch::Tensor& inout,
                                           const torch::Tensor& mask)
{
    CHECK_INPUT(inout);
    CHECK_INPUT(mask);

    T* inout_ptr = (T*)inout.data_ptr();
    const T* mask_ptr = (const T*)mask.data_ptr();

    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();
    Softmax<T> _softmax = Softmax<T>(typename Softmax<T>::Config(bsz, num_heads, seq_len));
    _softmax.SetSeqLength(seq_len);
    _softmax.Forward(bsz, inout_ptr, mask_ptr, q);
    return {inout};
}

template <typename T>
std::vector<torch::Tensor> softmax_backward(int bsz,
                                            int seq_len,
                                            int num_heads,
                                            torch::Tensor& out_grad,
                                            const torch::Tensor& input)
{
    CHECK_INPUT(out_grad);
    CHECK_INPUT(input);

    T* out_grad_ptr = (T*)out_grad.data_ptr();
    const T* input_ptr = (const T*)input.data_ptr();

    sycl::queue* q = ::SyclContext::Instance().GetCurrentStream();
    Softmax<T> _softmax = Softmax<T>(typename Softmax<T>::Config(bsz, num_heads, seq_len));
    _softmax.SetSeqLength(seq_len);

    _softmax.Backward(bsz, out_grad_ptr, input_ptr, q);
    return {out_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward_fp32", &softmax_forward<float>,
          "SOFTMAX forward with fp32 (DPCPP)");
    m.def("forward_bf16", &softmax_forward<bf16>,
          "SOFTMAX forward with bf16 (DPCPP)");
    m.def("forward_fp16", &softmax_forward<sycl::half>,
          "SOFTMAX forward with fp16 (DPCPP)");
    m.def("backward_fp32", &softmax_backward<float>,
          "SOFTMAX backward with fp32 (DPCPP)");
    m.def("backward_bf16", &softmax_backward<bf16>,
          "SOFTMAX backward with bf16 (DPCPP)");
    m.def("backward_fp16", &softmax_backward<sycl::half>,
          "SOFTMAX backward with fp16 (DPCPP)");
}
