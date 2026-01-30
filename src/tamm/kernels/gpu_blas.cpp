#include "tamm/utils.hpp"
#include "tamm_blas.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>

#if defined(USE_CUDA)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#elif defined(USE_HIP)
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#elif defined(USE_DPCPP)
#include <oneapi/mkl/blas.hpp>
#if defined(USE_PORT_BLAS)
#include <portblas.hpp>
#endif // USE_PORT_BLAS
#endif // USE_DPCPP

// =============================================================================
// GEMM Profiler - CSV Logger for CUDA kernel timing
// =============================================================================
namespace tamm::kernels::gpu {

enum class GemmDType {
  FP64,   // double precision (default)
  FP32,   // single precision
  FP16,   // half precision
  BF16,   // bfloat16
  TF32    // TensorFloat-32 (uses FP32 storage with TF32 compute)
};

class GemmProfiler {
private:
  std::ofstream csv_file_;
  std::mutex    mutex_;
  bool          enabled_ = false;
  bool          initialized_ = false;
  GemmDType     dtype_ = GemmDType::FP64;
  std::string   dtype_str_ = "fp64";

  GemmProfiler() {
    // Check if profiling is enabled via environment variable
    const char* profile_env = std::getenv("TAMM_GEMM_PROFILE");
    if(profile_env && std::string(profile_env) == "1") {
      enabled_ = true;
    }

    // Get dtype from environment variable
    const char* dtype_env = std::getenv("TAMM_GEMM_DTYPE");
    if(dtype_env) {
      std::string dtype_val(dtype_env);
      if(dtype_val == "fp64" || dtype_val == "FP64") {
        dtype_ = GemmDType::FP64;
        dtype_str_ = "fp64";
      }
      else if(dtype_val == "fp32" || dtype_val == "FP32") {
        dtype_ = GemmDType::FP32;
        dtype_str_ = "fp32";
      }
      else if(dtype_val == "fp16" || dtype_val == "FP16") {
        dtype_ = GemmDType::FP16;
        dtype_str_ = "fp16";
      }
      else if(dtype_val == "bf16" || dtype_val == "BF16") {
        dtype_ = GemmDType::BF16;
        dtype_str_ = "bf16";
      }
      else if(dtype_val == "tf32" || dtype_val == "TF32") {
        dtype_ = GemmDType::TF32;
        dtype_str_ = "tf32";
      }
    }

    // Get CSV output path
    const char* csv_path = std::getenv("TAMM_GEMM_CSV");
    std::string csv_filename = csv_path ? std::string(csv_path) : "tamm_gemm_profile.csv";

    if(enabled_) {
      csv_file_.open(csv_filename, std::ios::out | std::ios::trunc);
      if(csv_file_.is_open()) {
        csv_file_ << "m,n,k,dtype,time_us,gflops\n";
        csv_file_.flush();
        initialized_ = true;
      }
    }
  }

public:
  static GemmProfiler& instance() {
    static GemmProfiler profiler;
    return profiler;
  }

  bool is_enabled() const { return enabled_ && initialized_; }

  GemmDType get_dtype() const { return dtype_; }
  const std::string& get_dtype_str() const { return dtype_str_; }

  void log(int m, int n, int k, double time_us) {
    if(!is_enabled()) return;

    std::lock_guard<std::mutex> lock(mutex_);

    // Calculate GFLOPS: 2*m*n*k operations for GEMM
    double gflops = (2.0 * m * n * k) / (time_us * 1000.0);  // time_us to seconds, ops to GFLOPS

    csv_file_ << m << "," << n << "," << k << ","
              << dtype_str_ << "," << time_us << "," << gflops << "\n";
    csv_file_.flush();
  }

  ~GemmProfiler() {
    if(csv_file_.is_open()) {
      csv_file_.close();
    }
  }

  GemmProfiler(const GemmProfiler&) = delete;
  GemmProfiler& operator=(const GemmProfiler&) = delete;
};

} // namespace tamm::kernels::gpu

#if defined(USE_DPCPP)
#define ONEMKLBLAS_CHECK(FUNC)                                                         \
  do {                                                                                 \
    try {                                                                              \
      (FUNC);                                                                          \
    } catch(oneapi::mkl::exception const& ex) {                                        \
      std::ostringstream msg;                                                          \
      msg << "oneMKL Error: " << ex.what() << ", at " << __FILE__ << " : " << __LINE__ \
          << std::endl;                                                                \
      throw std::runtime_error(msg.str());                                             \
    }                                                                                  \
  } while(0)
#endif // USE_DPCPP

template<typename T>
void tamm::kernels::gpu::axpy(const int64_t n, const T* src, const int incx, T*& dst,
                              const int incy, gpuStream_t& handle) {
  T alpha = 1.0;
#if defined(USE_DPCPP)
  ONEMKLBLAS_CHECK(
    oneapi::mkl::blas::column_major::axpy(handle.first, n, alpha, src, incx, dst, incy));
#elif defined(USE_CUDA)
  CUBLAS_CHECK(cublasDaxpy(handle.second, n, &alpha, src, incx, dst, incy));
#elif defined(USE_HIP)
  ROCBLAS_CHECK(rocblas_daxpy(handle.second, n, &alpha, src, incx, dst, incy));
#endif
}

template<typename T, typename T1, typename T2, typename T3>
void tamm::kernels::gpu::gemm(int n, int m, int k, const T alpha, const T3* B, int ldb, const T2* A,
                              int lda, const T beta, T1* C, int ldc, gpuStream_t& handle) {
#if defined(USE_DPCPP)

#ifdef USE_PORT_BLAS
  blas::SB_Handle sb_handle(handle.first);
  blas::internal::_gemm(sb_handle, 'n', 'n', n, m, k, alpha, const_cast<T3*>(B), ldb,
                        const_cast<T2*>(A), lda, beta, C, ldc, {});
  handle.first.wait();
#else
  auto gemm_event = oneapi::mkl::blas::column_major::gemm(handle.first, oneapi::mkl::transpose::N,
                                                          oneapi::mkl::transpose::N, n, m, k, alpha,
                                                          B, ldb, A, lda, beta, C, ldc);
  gemm_event.wait();
#endif // USE_PORT_BLAS

#elif defined(USE_CUDA)
  auto& profiler = GemmProfiler::instance();
  GemmDType dtype = profiler.get_dtype();

  // Synchronize before timing if profiling enabled
  if(profiler.is_enabled()) {
    cudaDeviceSynchronize();
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  if constexpr(tamm::internal::is_complex_v<T1> && tamm::internal::is_complex_v<T2> &&
               tamm::internal::is_complex_v<T3>) {
    // Complex types - always use ZGEMM (FP64 complex)
    CUBLAS_CHECK(cublasZgemm(handle.second, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             (cuDoubleComplex*) &alpha, (cuDoubleComplex*) B, ldb,
                             (cuDoubleComplex*) A, lda, (cuDoubleComplex*) &beta,
                             (cuDoubleComplex*) C, ldc));
  }
  else {
    // Real types - switch based on TAMM_GEMM_DTYPE environment variable
    switch(dtype) {
      case GemmDType::TF32: {
        // TF32: Use cublasGemmEx with TF32 compute type (requires Ampere+)
        // Input/output remain FP32, but compute uses TF32
        cublasSetMathMode(handle.second, CUBLAS_TF32_TENSOR_OP_MATH);
        float alpha_f = static_cast<float>(alpha);
        float beta_f = static_cast<float>(beta);
        CUBLAS_CHECK(cublasGemmEx(handle.second, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                  &alpha_f,
                                  B, CUDA_R_64F, ldb,
                                  A, CUDA_R_64F, lda,
                                  &beta_f,
                                  C, CUDA_R_64F, ldc,
                                  CUBLAS_COMPUTE_32F_FAST_TF32,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        cublasSetMathMode(handle.second, CUBLAS_DEFAULT_MATH);
        break;
      }
      case GemmDType::FP32: {
        // FP32: Use cublasGemmEx with FP32 compute
        float alpha_f = static_cast<float>(alpha);
        float beta_f = static_cast<float>(beta);
        CUBLAS_CHECK(cublasGemmEx(handle.second, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                  &alpha_f,
                                  B, CUDA_R_64F, ldb,
                                  A, CUDA_R_64F, lda,
                                  &beta_f,
                                  C, CUDA_R_64F, ldc,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
        break;
      }
      case GemmDType::FP16: {
        // FP16: Use cublasGemmEx with FP16 compute (data stays FP64, compute in FP16)
        float alpha_f = static_cast<float>(alpha);
        float beta_f = static_cast<float>(beta);
        CUBLAS_CHECK(cublasGemmEx(handle.second, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                  &alpha_f,
                                  B, CUDA_R_64F, ldb,
                                  A, CUDA_R_64F, lda,
                                  &beta_f,
                                  C, CUDA_R_64F, ldc,
                                  CUBLAS_COMPUTE_16F,
                                  CUBLAS_GEMM_DEFAULT));
        break;
      }
      case GemmDType::BF16: {
        // BF16: Use cublasGemmEx with BF16 compute (requires Ampere+)
        float alpha_f = static_cast<float>(alpha);
        float beta_f = static_cast<float>(beta);
        cublasSetMathMode(handle.second, CUBLAS_TF32_TENSOR_OP_MATH);
        CUBLAS_CHECK(cublasGemmEx(handle.second, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                  &alpha_f,
                                  B, CUDA_R_64F, ldb,
                                  A, CUDA_R_64F, lda,
                                  &beta_f,
                                  C, CUDA_R_64F, ldc,
                                  CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        cublasSetMathMode(handle.second, CUBLAS_DEFAULT_MATH);
        break;
      }
      case GemmDType::FP64:
      default: {
        // FP64: Default double precision GEMM
        CUBLAS_CHECK(cublasDgemm(handle.second, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, ldb, A,
                                 lda, &beta, C, ldc));
        break;
      }
    }
  }

  // Synchronize and record timing if profiling enabled
  if(profiler.is_enabled()) {
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    double time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    profiler.log(m, n, k, time_us);
  }
#elif defined(USE_HIP)
  if constexpr(internal::is_complex_v<T1> && internal::is_complex_v<T2> &&
               internal::is_complex_v<T3>) {
    ROCBLAS_CHECK(rocblas_zgemm(handle.second, rocblas_operation_none, rocblas_operation_none, n, m,
                                k, (rocblas_double_complex*) &alpha, (rocblas_double_complex*) B,
                                ldb, (rocblas_double_complex*) A, lda,
                                (rocblas_double_complex*) &beta, (rocblas_double_complex*) C, ldc));
  }
  else {
    ROCBLAS_CHECK(rocblas_dgemm(handle.second, rocblas_operation_none, rocblas_operation_none, n, m,
                                k, &alpha, B, ldb, A, lda, &beta, C, ldc));
  }
#endif
}

template void tamm::kernels::gpu::axpy(const int64_t n, const double* src, const int incx,
                                       double*& dst, const int incy, gpuStream_t& thandle);

template void tamm::kernels::gpu::gemm(int n, int m, int k, const double alpha, const double* B,
                                       int ldb, const double* A, int lda, const double beta,
                                       double* C, int ldc, gpuStream_t& handle);
#if !defined(USE_PORT_BLAS)
template void tamm::kernels::gpu::gemm(int n, int m, int k, const std::complex<double> alpha,
                                       const std::complex<double>* B, int ldb,
                                       const std::complex<double>* A, int lda,
                                       const std::complex<double> beta, std::complex<double>* C,
                                       int ldc, gpuStream_t& handle);
#endif // USE_PORT_BLAS
