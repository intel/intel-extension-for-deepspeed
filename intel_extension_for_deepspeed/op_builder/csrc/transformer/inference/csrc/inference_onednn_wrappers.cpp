#include "inference_onednn_wrappers.hpp"
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <ATen/ATen.h>
#include "inference_sycl_layers.h"

template <typename T, bool bmm>
inline int onednn_matmul(sycl::queue handle,
                         bool trans_src,
                         bool trans_wgt,
                         int m,
                         int n,
                         int k,
                         const float alpha,
                         const float beta,
                         const T* src_ptr,
                         const T* wgt_ptr,
                         T* dst_ptr,
                         int batch)
{
    /*
     * src, [m, k], m: batch, k: in_feature
     * wgt, [k, n], n: k: in_features, out_feature
     * dst, [m, n], m: batch, n: out_features
     */
    device dev = handle.get_device();
    context ctx = handle.get_context();
    dnnl::engine engine = dnnl::sycl_interop::make_engine(dev, ctx);
    dnnl::stream stream = dnnl::sycl_interop::make_stream(engine, handle);

    dnnl::memory::dims src_dims, wgt_dims, dst_dims;
    constexpr auto dnnl_dtype_16 = std::is_same<T, fp16>::value ? dnnl::memory::data_type::f16
                                                                : dnnl::memory::data_type::bf16;
    if constexpr (bmm) {
        src_dims = {batch, m, k};
        wgt_dims = {batch, k, n};
        dst_dims = {batch, m, n};
    } else {
        src_dims = {m, k};
        wgt_dims = {k, n};
        dst_dims = {m, n};
    }

    dnnl::memory::desc src_md, wgt_md, dst_md;

    if constexpr (bmm) {
        src_md = dnnl::memory::desc(
            src_dims,
            dnnl_dtype_16,
            trans_src ? dnnl::memory::format_tag::acb : dnnl::memory::format_tag::abc);
        wgt_md = dnnl::memory::desc(
            wgt_dims,
            dnnl_dtype_16,
            trans_wgt ? dnnl::memory::format_tag::acb : dnnl::memory::format_tag::abc);
        dst_md = dnnl::memory::desc(dst_dims, dnnl_dtype_16, dnnl::memory::format_tag::abc);
    } else {
        src_md = dnnl::memory::desc(
            src_dims,
            dnnl_dtype_16,
            trans_src ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
        wgt_md = dnnl::memory::desc(
            wgt_dims,
            dnnl_dtype_16,
            trans_wgt ? dnnl::memory::format_tag::ba : dnnl::memory::format_tag::ab);
        dst_md = dnnl::memory::desc(dst_dims, dnnl_dtype_16, dnnl::memory::format_tag::ab);
    }

    auto src_mem = dnnl::memory(src_md, engine, (void*)src_ptr);
    auto wgt_mem = dnnl::memory(wgt_md, engine, (void*)wgt_ptr);
    auto dst_mem = dnnl::memory(dst_md, engine, (void*)dst_ptr);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    std::unordered_map<int, dnnl::memory> matmul_args;
    if (alpha != 1.0f) {
        float alpha_v(alpha);
        attr.set_scales_mask(DNNL_ARG_DST, /* mask */ 0);
        dnnl::memory alpha_mem({{1}, dnnl::memory::data_type::f32, {1}}, engine, &alpha_v);
        matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, alpha_mem});
    }
    if (beta != 0.0f) {
        dnnl::post_ops po;
        po.append_sum(beta);
        attr.set_post_ops(po);
    }

    auto matmul_pd = dnnl::matmul::primitive_desc(engine, src_md, wgt_md, dst_md, attr);

    auto matmul_prim = dnnl::matmul(matmul_pd);
    dnnl::memory::desc scratchpad_md = matmul_pd.scratchpad_desc();
    auto options = at::TensorOptions()
                       .dtype(at::kByte)
                       .layout(at::kStrided)
                       .device(at::kXPU)
                       .requires_grad(false);
    auto scratchpad_tensor = at::empty({(int64_t)scratchpad_md.get_size()}, options);
    dnnl::memory scratchpad(scratchpad_md, engine, scratchpad_tensor.data_ptr());

    matmul_args.insert({DNNL_ARG_SRC, src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, wgt_mem});
    matmul_args.insert({DNNL_ARG_DST, dst_mem});
    matmul_args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});

    matmul_prim.execute(stream, matmul_args);
    stream.wait();
}

template <typename T>
int onednn_matmul_ex(sycl::queue handle,
                     bool trans_src,
                     bool trans_wgt,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const T* src_ptr,
                     const T* wgt_ptr,
                     T* dst_ptr)
{
    onednn_matmul<T, false>(
        handle, trans_src, trans_wgt, m, n, k, alpha, beta, src_ptr, wgt_ptr, dst_ptr, 1);
}

template <typename T>
int onednn_batchgemm(sycl::queue handle,
                     int m,
                     int n,
                     int k,
                     const float alpha,
                     const float beta,
                     const T* src_ptr,
                     const T* wgt_ptr,
                     T* dst_ptr,
                     bool trans_src,
                     bool trans_wgt,
                     int batch)
{
    onednn_matmul<T, true>(
        handle, trans_src, trans_wgt, m, n, k, alpha, beta, src_ptr, wgt_ptr, dst_ptr, batch);
}

template int onednn_matmul_ex(sycl::queue handle,
                              bool trans_src,
                              bool trans_wgt,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const bf16* src_ptr,
                              const bf16* wgt_ptr,
                              bf16* dst_ptr);

template int onednn_matmul_ex(sycl::queue handle,
                              bool trans_src,
                              bool trans_wgt,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const fp16* src_ptr,
                              const fp16* wgt_ptr,
                              fp16* dst_ptr);

template int onednn_batchgemm(sycl::queue handle,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const bf16* src_ptr,
                              const bf16* wgt_ptr,
                              bf16* dst_ptr,
                              bool trans_src,
                              bool trans_wgt,
                              int batch);

template int onednn_batchgemm(sycl::queue handle,
                              int m,
                              int n,
                              int k,
                              const float alpha,
                              const float beta,
                              const fp16* src_ptr,
                              const fp16* wgt_ptr,
                              fp16* dst_ptr,
                              bool trans_src,
                              bool trans_wgt,
                              int batch);