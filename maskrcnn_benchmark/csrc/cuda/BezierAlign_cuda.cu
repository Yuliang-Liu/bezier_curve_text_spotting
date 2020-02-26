// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                   \
       i += blockDim.x * gridDim.x)

template <typename T> __device__ T BEZIER_CURVE(T p0, T p1, T p2, T p3, const T u) {
  return ((1. - u) * (1. - u) * (1. - u) * p0 +
          3. * u * (1. - u) * (1. - u) * p1 + 3. * u * u * (1. - u) * p2 +
          u * u * u * p3);
}

/**
 * bilinear_interpolate
 * sample one point at (x, y) from bottom_data
 * @param bottom_data array of feature map
 * @param height, width size of feature map
 * @param y, x sample location
 */
template <typename T>
__device__ T bilinear_interpolate(const T *bottom_data, const int height,
                                  const int width, T y, T x,
                                  const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void
BezierAlignForward(const int nthreads, const T *bottom_data,
                   const T spatial_scale, const int channels, const int height,
                   const int width, const int pooled_height,
                   const int pooled_width, const int sampling_ratio,
                   const T *bottom_beziers, T *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // beziers have size Nx(1+8x2)
    const T *offset_bottom_beziers = bottom_beziers + n * (1 + 8 * 2);
    int bezier_batch_ind = offset_bottom_beziers[0];

    // Do not using rounding; this implementation detail is critical
    T p0_y = offset_bottom_beziers[1] * spatial_scale;
    T p0_x = offset_bottom_beziers[2] * spatial_scale;
    T p1_y = offset_bottom_beziers[3] * spatial_scale;
    T p1_x = offset_bottom_beziers[4] * spatial_scale;
    T p2_y = offset_bottom_beziers[5] * spatial_scale;
    T p2_x = offset_bottom_beziers[6] * spatial_scale;
    T p3_y = offset_bottom_beziers[7] * spatial_scale;
    T p3_x = offset_bottom_beziers[8] * spatial_scale;
    T p4_y = offset_bottom_beziers[9] * spatial_scale;
    T p4_x = offset_bottom_beziers[10] * spatial_scale;
    T p5_y = offset_bottom_beziers[11] * spatial_scale;
    T p5_x = offset_bottom_beziers[12] * spatial_scale;
    T p6_y = offset_bottom_beziers[13] * spatial_scale;
    T p6_x = offset_bottom_beziers[14] * spatial_scale;
    T p7_y = offset_bottom_beziers[15] * spatial_scale;
    T p7_x = offset_bottom_beziers[16] * spatial_scale;

    const T *offset_bottom_data =
        bottom_data + (bezier_batch_ind * channels + c) * height * width;

    // compute the coords
    const T u = pw / static_cast<T>(pooled_width);
    const T v = ph / static_cast<T>(pooled_height);
    const T y0 = BEZIER_CURVE(p0_y, p1_y, p2_y, p3_y, u);
    const T x0 = BEZIER_CURVE(p0_x, p1_x, p2_x, p3_x, u);
    const T y1 = BEZIER_CURVE(p4_y, p5_y, p6_y, p7_y, u);
    const T x1 = BEZIER_CURVE(p4_x, p5_x, p6_x, p7_x, u);
    const T y = y1 * v + y0 * (1. - v);
    const T x = x1 * v + x0 * (1. - v);

    top_data[index] =
        bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
  }
}

template <typename T>
__device__ void
bilinear_interpolate_gradient(const int height, const int width, T y, T x,
                              T &w1, T &w2, T &w3, T &w4, int &x_low,
                              int &x_high, int &y_low, int &y_high,
                              const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename T>
__global__ void BezierAlignBackwardFeature(
    const int nthreads, const T *top_diff, const int num_beziers,
    const T spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int sampling_ratio, T *bottom_diff, const T *bottom_beziers) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T *offset_bottom_beziers = bottom_beziers + n * (1 + 8 * 2);
    int bezier_batch_ind = offset_bottom_beziers[0];

    // Do not using rounding; this implementation detail is critical
    T p0_y = offset_bottom_beziers[1] * spatial_scale;
    T p0_x = offset_bottom_beziers[2] * spatial_scale;
    T p1_y = offset_bottom_beziers[3] * spatial_scale;
    T p1_x = offset_bottom_beziers[4] * spatial_scale;
    T p2_y = offset_bottom_beziers[5] * spatial_scale;
    T p2_x = offset_bottom_beziers[6] * spatial_scale;
    T p3_y = offset_bottom_beziers[7] * spatial_scale;
    T p3_x = offset_bottom_beziers[8] * spatial_scale;
    T p4_y = offset_bottom_beziers[9] * spatial_scale;
    T p4_x = offset_bottom_beziers[10] * spatial_scale;
    T p5_y = offset_bottom_beziers[11] * spatial_scale;
    T p5_x = offset_bottom_beziers[12] * spatial_scale;
    T p6_y = offset_bottom_beziers[13] * spatial_scale;
    T p6_x = offset_bottom_beziers[14] * spatial_scale;
    T p7_y = offset_bottom_beziers[15] * spatial_scale;
    T p7_x = offset_bottom_beziers[16] * spatial_scale;

    T *offset_bottom_diff =
        bottom_diff + (bezier_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T *offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    T w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;

    // compute the coords
    const T u = pw / static_cast<T>(pooled_width);
    const T v = ph / static_cast<T>(pooled_height);
    const T y0 = BEZIER_CURVE(p0_y, p1_y, p2_y, p3_y, u);
    const T x0 = BEZIER_CURVE(p0_x, p1_x, p2_x, p3_x, u);
    const T y1 = BEZIER_CURVE(p4_y, p5_y, p6_y, p7_y, u);
    const T x1 = BEZIER_CURVE(p4_x, p5_x, p6_x, p7_x, u);
    const T y = y1 * v + y0 * (1. - v);
    const T x = x1 * v + x0 * (1. - v);

    bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, x_low,
                                  x_high, y_low, y_high, index);

    if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
      atomicAdd(offset_bottom_diff + y_low * width + x_low,
                static_cast<T>(top_diff_this_bin * w1));
      atomicAdd(offset_bottom_diff + y_low * width + x_high,
                static_cast<T>(top_diff_this_bin * w2));
      atomicAdd(offset_bottom_diff + y_high * width + x_low,
                static_cast<T>(top_diff_this_bin * w3));
      atomicAdd(offset_bottom_diff + y_high * width + x_high,
                static_cast<T>(top_diff_this_bin * w4));
    }
  } // CUDA_1D_KERNEL_LOOP
} // BezierAlignBackward

at::Tensor
BezierAlign_forward_cuda(const at::Tensor &input, const at::Tensor &beziers,
                         const float spatial_scale, const int pooled_height,
                         const int pooled_width, const int sampling_ratio) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(beziers.type().is_cuda(), "beziers must be a CUDA tensor");

  auto num_beziers = beziers.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty({num_beziers, channels, pooled_height, pooled_width},
                          input.options());
  auto output_size = num_beziers * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "BezierAlign_forward", [&] {
    BezierAlignForward<scalar_t><<<grid, block, 0, stream>>>(
        output_size, input.contiguous().data<scalar_t>(), spatial_scale,
        channels, height, width, pooled_height, pooled_width, sampling_ratio,
        beziers.contiguous().data<scalar_t>(), output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor
BezierAlign_backward_cuda(const at::Tensor &grad, const at::Tensor &beziers,
                          const float spatial_scale, const int pooled_height,
                          const int pooled_width, const int batch_size,
                          const int channels, const int height, const int width,
                          const int sampling_ratio) {
  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(beziers.type().is_cuda(), "beziers must be a CUDA tensor");

  auto num_beziers = beziers.size(0);
  auto grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "BezierAlign_backward", [&] {
    BezierAlignBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
        grad.numel(), grad.contiguous().data<scalar_t>(), num_beziers,
        spatial_scale, channels, height, width, pooled_height, pooled_width,
        sampling_ratio, grad_input.data<scalar_t>(),
        beziers.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return grad_input;
}
