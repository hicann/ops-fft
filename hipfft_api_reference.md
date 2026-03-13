# hipFFT API 接口总结

本文档总结了 hipFFT 仓库对外暴露的所有 API 接口。

---

## 1. 计划创建函数 (Plan Creation)

| 函数 | 说明 | 绑定类型 |
|------|------|---------|
| `hipfftPlan1d()` | 创建一维 FFT 计划 | C |
| `hipfftPlan2d()` | 创建二维 FFT 计划 | C |
| `hipfftPlan3d()` | 创建三维 FFT 计划 | C |
| `hipfftPlanMany()` | 创建批量多维 FFT 计划 | C |

### 函数签名

```c
// 一维 FFT 计划
hipfftResult hipfftPlan1d(hipfftHandle *plan, int nx, hipfftType type, int batch);

// 二维 FFT 计划
hipfftResult hipfftPlan2d(hipfftHandle *plan, int nx, int ny, hipfftType type, int batch);

// 三维 FFT 计划
hipfftResult hipfftPlan3d(hipfftHandle *plan, int nx, int ny, int nz, hipfftType type, int batch);
```

### 参数说明

| 参数 | 方向 | 描述 |
|------|------|------|
| `plan` | [out] | 指向 FFT 计划句柄的指针 |
| `nx/ny/nz` | [in] | FFT 在各维度上的长度（变换的大小） |
| `type` | [in] | FFT 类型（复数到复数、实数到复数等） |
| `batch` | [in] | 批量计算的变换数量 |

---

## 2. 执行函数 (Execution Functions)

hipFFT 提供了多种数据类型的执行函数：

### 2.1 单精度 (float/complex-float)

| 函数 | 说明 | 变换类型 |
|------|------|---------|
| `hipfftExecC2C()` | 复数到复数 | Complex-to-Complex |
| `hipfftExecR2C()` | 实数到复数 | Real-to-Complex |
| `hipfftExecC2R()` | 复数到实数 | Complex-to-Real |

### 2.2 双精度

| 函数 | 说明 | 变换类型 |
|------|------|---------|
| `hipfftExecZ2Z()` | 双精度复数到复数 | Double Complex-to-Complex |
| `hipfftExecD2Z()` | 双精度实数到复数 | Double Real-to-Complex |
| `hipfftExecZ2D()` | 双精度复数到实数 | Double Complex-to-Real |

### 函数签名

```c
// 复数到复数
hipfftResult hipfftExecC2C(hipfftHandle plan, hipfftComplex *idata,
                            hipfftComplex *odata, int direction);

// 实数到复数
hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal *idata,
                            hipfftComplex *odata, int direction);

// 复数到实数
hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex *idata,
                            hipfftReal *odata, int direction);

// 双精度复数到复数
hipfftResult hipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex *idata,
                            hipfftDoubleComplex *odata, int direction);

// 双精度实数到复数
hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal *idata,
                            hipfftDoubleComplex *odata, int direction);

// 双精度复数到实数
hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex *idata,
                            hipfftDoubleReal *odata, int direction);
```

---

## 3. 资源管理函数 (Resource Management)

| 函数 | 说明 |
|------|------|
| `hipfftCreate()` | 创建 hipFFT 句柄 |
| `hipfftDestroy()` | 销毁 FFT 计划并释放资源 |
| `hipfftSetStream()` | 设置 HIP 流 |
| `hipfftGetStream()` | 获取当前关联的 HIP 流 |

### 函数签名

```c
// 创建句柄
hipfftResult hipfftCreate(hipfftHandle *plan);

// 销毁计划
hipfftResult hipfftDestroy(hipfftHandle plan);

// 设置流
hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream);

// 获取流
hipfftResult hipfftGetStream(hipfftHandle plan, hipStream_t *stream);
```

---

## 4. 多 GPU 扩展函数 (Multi-GPU)

| 函数 | 说明 |
|------|------|
| `hipfftXt` 系列 | 跨多个 GPU 执行 FFT 的扩展 API |

**说明：** hipfftXt 提供了扩展版本，用于在多个 GPU 上执行 FFT，自动管理多 GPU 计划。

---

## 5. 数据类型 (Data Types)

| 类型 | 说明 |
|------|------|
| `hipfftHandle` | FFT 计划句柄 |
| `hipfftReal` | 实数类型 (float) |
| `hipfftDoubleReal` | 双精度实数类型 (double) |
| `hipfftComplex` | 单精度复数 |
| `hipfftDoubleComplex` | 双精度复数 |
| `hipfftType` | FFT 变换类型枚举 |
| `hipStream_t` | HIP 流类型 |

### 5.1 hipfftComplex/hipfftReal 类型定义

hipFFT 的数据类型是对 HIP 运行时类型的封装：

```c
// hipfft.h 中的定义
typedef hipComplex hipfftComplex;           // 单精度复数 = HIP运行时类型
typedef hipDoubleComplex hipfftDoubleComplex; // 双精度复数 = HIP运行时类型
typedef float hipfftReal;                   // 单精度实数
typedef double hipfftDoubleReal;            // 双精度实数
```

**hipComplex 底层定义**（来自 HIP 运行时库，非 hipFFT 源码）：

```c
// 位于 HIP 运行时库 ($ROCM_PATH/hip/include/hip/hip_vector_types.h)
struct hipComplex {
    float x;   // 实部
    float y;   // 虚部
};

struct hipDoubleComplex {
    double x;  // 实部
    double y;  // 虚部
};
```

**类型对应关系**：

| hipFFT 类型 | HIP 底层类型 | 大小 | 组成 |
|------------|-------------|------|------|
| `hipfftReal` | `float` | 4字节 | 实数 |
| `hipfftDoubleReal` | `double` | 8字节 | 实数 |
| `hipfftComplex` | `hipComplex` | 8字节 | float x + float y |
| `hipfftDoubleComplex` | `hipDoubleComplex` | 16字节 | double x + double y |

**变换类型与数据类型对应**：

| 变换类型 | 输入类型 | 输出类型 |
|---------|---------|---------|
| `HIPFFT_R2C` | `hipfftReal` | `hipfftComplex` |
| `HIPFFT_C2R` | `hipfftComplex` | `hipfftReal` |
| `HIPFFT_C2C` | `hipfftComplex` | `hipfftComplex` |
| `HIPFFT_D2Z` | `hipfftDoubleReal` | `hipfftDoubleComplex` |
| `HIPFFT_Z2D` | `hipfftDoubleComplex` | `hipfftDoubleReal` |
| `HIPFFT_Z2Z` | `hipfftDoubleComplex` | `hipfftDoubleComplex` |

### 5.2 hipfftHandle_t 结构体（AMD路径）

| 分类 | 成员 | 用处 |
|------|------|------|
| **类型信息** | `hipfftIOType type` (inputType, outputType) | 记录输入/输出数据类型(C2C/R2C/C2R等)，选择正确的rocFFT计划 |
| **rocFFT计划** ⚠️ | `ip_forward`, `op_forward`, `ip_inverse`, `op_inverse` | 为兼容cuFFT Exec API需预创建4个计划，Exec时通过指针比较和direction动态选择 |
| **执行信息 & 工作缓冲区** | `info`, `workBuffer`, `workBufferSize`, `autoAllocate`, `workBufferNeedsFree` | info存储流/回调信息；workBuffer为FFT算法临时存储区 |
| **Callback 支持** | `load_callback_*`, `store_callback_*` (6个成员) | 用户自定义加载/存储回调函数，在FFT前后对数据做预处理/后处理 |
| **数据布局信息** | `inLength`, `outLength`, `ionembed`, `inStrides`, `outStrides`, `iDist`, `oDist`, `batch`, `scale_factor` | 描述多维数据维度/步长/距离，支持非连续内存布局和批量变换 |
| **多GPU / MPI 支持** | `inBricks`, `outBricks`, `singleProcMultiDevice`, `comm_type`, `comm_handle` | bricks分解数据到多GPU；MPI支持分布式FFT跨进程通信 |

> ⚠️ **设计说明**：为兼容 cuFFT 的 `hipfftExecC2C(plan, idata, odata, direction)` API（执行时通过 `idata==odata` 判断 placement，通过 `direction` 参数确定方向），必须预创建 4 个 rocFFT 计划覆盖所有组合，导致约 **4x** 的内存和创建时间开销。

### 5.2 rocfft_execution_info_t 结构体

hipfftHandle_t 中的 `info` 成员指向 `rocfft_execution_info_t` 结构体，用于存储 FFT 执行时的运行时信息：

| 成员 | 类型 | 用处 |
|------|------|------|
| `workBuffer` | `void*` | FFT算法临时存储区（twiddle因子等） |
| `workBufferSize` | `size_t` | 工作缓冲区大小（字节） |
| `rocfft_stream` | `hipStream_t` | 执行FFT的HIP流，支持异步执行 |
| `load_cb_fns` | `void**` | 加载回调函数指针数组（每个brick一个） |
| `load_cb_data` | `void**` | 加载回调的用户数据数组 |
| `load_cb_lds_bytes` | `size_t` | 加载回调可用的共享内存大小 |
| `store_cb_fns` | `void**` | 存储回调函数指针数组 |
| `store_cb_data` | `void**` | 存储回调的用户数据数组 |
| `store_cb_lds_bytes` | `size_t` | 存储回调可用的共享内存大小 |

> **说明**：`rocfft_execution_info_t` 不包含 FFT 计划本身，只存储执行时需要的临时信息（工作缓冲区、HIP流、回调函数）。

### FFT 类型枚举 (hipfftType)

| 枚举值 | 说明 |
|--------|------|
| `HIPFFT_C2C` | Complex to Complex - 复数到复数 |
| `HIPFFT_R2C` | Real to Complex - 实数到复数 |
| `HIPFFT_C2R` | Complex to Real - 复数到实数 |
| `HIPFFT_D2Z` | Double Real to Complex - 双精度实数到复数 |
| `HIPFFT_Z2D` | Double Complex to Real - 双精度复数到实数 |
| `HIPFFT_Z2Z` | Double Complex to Complex - 双精度复数到复数 |

---

## 6. 方向参数 (Direction Parameters)

| 常量 | 值 | 说明 |
|------|---|------|
| `HIPFFT_FORWARD` | -1 | 前向 FFT 变换 |
| `HIPFFT_INVERSE` | 1 | 逆 FFT 变换 |

---

## 7. 核心参数详解 (Batch / Direction / Placement)

### 7.1 Batch（批量）

**作用**：一次执行多个相同大小的 FFT 变换

```
batch = 1（单个FFT）:
  输入: [x0, x1, x2, ..., xN-1]     N点
  输出: [X0, X1, X2, ..., XN-1]     N点

batch = 3（批量FFT）:
  输入:  [x0...xN-1] [x0...xN-1] [x0...xN-1]
          ↑ batch 0    ↑ batch 1    ↑ batch 2
  输出:  [X0...XN-1] [X0...XN-1] [X0...XN-1]
          ↑ batch 0    ↑ batch 1    ↑ batch 2
```

| 应用场景 | 说明 |
|---------|------|
| 图像处理 | 同时处理多行/多列的 FFT |
| 信号处理 | 同时分析多个信号通道 |
| 深度学习 | 批量数据的频域变换 |
| 卷积运算 | 多个卷积核的频域计算 |

---

### 7.2 Direction（方向）

**作用**：决定是前向 FFT 还是逆向 FFT

| 方向 | 变换 | 公式 | 应用 |
|------|------|------|------|
| **FORWARD** | 时域 → 频域 | X[k] = Σ x[n] * e^(-j*2π*n*k/N) | 频谱分析、滤波器设计 |
| **BACKWARD** | 频域 → 时域 | x[n] = (1/N) * Σ X[k] * e^(j*2π*n*k/N) | 信号重构、滤波后恢复 |

**典型应用流程**：
```
频域滤波: 时域 → FFT(FORWARD) → 频域滤波 → IFFT(BACKWARD) → 时域
快速卷积: 信号 FFT + 卷积核 FFT → 点乘 → IFFT → 结果
```

---

### 7.3 Placement（数据放置）

**作用**：决定输入输出是否使用同一块内存

| Placement | 说明 | 特点 |
|-----------|------|------|
| **OUT-OF-PLACE** | 输入和输出使用不同内存 | 输入数据保留，需要额外内存 |
| **IN-PLACE** | 输入和输出使用同一块内存 | 节省内存，但输入数据被覆盖 |

**hipFFT 中判断方式**：
```cpp
// 通过指针比较判断 placement
hipfftExecC2C(plan, data, data, direction);      // in == out → IN-PLACE
hipfftExecC2C(plan, input, output, direction);   // in != out → OUT-OF-PLACE
```

| 应用场景 | Placement | 原因 |
|---------|-----------|------|
| 内存受限 | IN-PLACE | 节省一半内存 |
| 需要保留输入 | OUT-OF-PLACE | 后续还需用原始数据 |
| 多步处理 | IN-PLACE | 中间结果不需要保留 |
| 调试验证 | OUT-OF-PLACE | 方便对比输入输出 |

---

### 7.4 三者组合的典型应用

| 场景 | batch | direction | placement | 说明 |
|------|-------|-----------|-----------|------|
| 图像频域滤波 | 图像行数 | FWD → BWD | IN-PLACE | 每行独立FFT，节省内存 |
| 快速卷积 | 卷积数量 | FWD+FWD+BWD | OUT-OF-PLACE | 保留中间结果 |
| 深度学习 | mini-batch | 视操作定 | IN-PLACE | 内存敏感场景 |

---

## 8. 返回值 (Return Values)

| 返回值 | 说明 |
|--------|------|
| `HIPFFT_SUCCESS` | 操作成功 |
| `HIPFFT_INVALID_PLAN` | 无效的计划 |
| `HIPFFT_ALLOC_FAILED` | 内存分配失败 |
| `HIPFFT_INVALID_TYPE` | 无效的类型 |
| `HIPFFT_INVALID_VALUE` | 无效的参数值 |
| `HIPFFT_INTERNAL_ERROR` | 内部错误 |
| `HIPFFT_EXEC_FAILED` | 执行失败 |
| `HIPFFT_SETUP_FAILED` | 设置失败 |
| `HIPFFT_INVALID_SIZE` | 无效的大小 |
| `HIPFFT_UNALIGNED_DATA` | 数据未对齐 |
| `HIPFFT_UNSUPPORTED` | 不支持的操作 |
| `HIPFFT_NOT_IMPLEMENTED` | 未实现的功能 |

---

## 8. 典型使用流程

### 8.1 基本流程

```cpp
#include <hipfft/hipfft.h>

// 1. 创建句柄
hipfftHandle plan;
hipfftCreate(&plan);

// 2. 创建计划
hipfftPlan1d(&plan, N, HIPFFT_C2C, 1);

// 3. 执行 FFT
hipfftExecC2C(plan, idata, odata, HIPFFT_FORWARD);

// 4. 清理
hipfftDestroy(plan);
```

### 8.2 完整示例

```cpp
#include <hipfft/hipfft.h>
#include <hip/hip_runtime.h>
#include <cmath>

// FFT 长度
#define N 1024

int main() {
    // 1. 创建计划
    hipfftHandle plan;
    hipfftPlan1d(&plan, N, HIPFFT_C2C, 1);

    // 2. 分配设备内存
    hipfftComplex* d_data;
    hipMalloc(&d_data, N * sizeof(hipfftComplex));

    // 3. 拷贝数据到设备
    hipMemcpy(d_data, h_data, N * sizeof(hipfftComplex),
               hipMemcpyHostToDevice);

    // 4. 执行前向 FFT
    hipfftExecC2C(plan, d_data, d_data, HIPFFT_FORWARD);

    // 5. 拷贝结果回主机
    hipMemcpy(h_data, d_data, N * sizeof(hipfftComplex),
               hipMemcpyDeviceToHost);

    // 6. 执行逆 FFT
    hipfftExecC2C(plan, d_data, d_data, HIPFFT_INVERSE);

    // 7. 清理
    hipFree(d_data);
    hipfftDestroy(plan);

    return 0;
}
```

### 8.3 实数到复数变换 (R2C)

```cpp
// 创建 R2C 计划
hipfftPlan1d(&plan, N, HIPFFT_R2C, 1);

// 实数输入
hipfftReal* input;
hipfftComplex* output;

// 执行变换
hipfftExecR2C(plan, input, output, HIPFFT_FORWARD);
```

---

## 9. API 特性

### 9.1 兼容性

- **CUDA 兼容**：hipFFT API 完全遵循 NVIDIA CUDA cuFFT API 规范
- **后端支持**：
  - rocFFT（AMD GPU）
  - cuFFT（NVIDIA GPU）
- **可移植性**：无需修改客户端代码即可在不同 GPU 平台运行

### 9.2 设计目标

hipFFT 是一个 FFT 调度库（marshalling library），提供：
- 薄包装层（thin wrapper）
- API 兼容性
- 平台无关的接口

---

## 10. 相关资源

### 10.1 官方文档

- [hipFFT Documentation (ROCm 7.1.0)](https://rocm.docs.amd.com/projects/hipFFT/en/docs-7.1.0/)
- [hipFFT API Usage](https://rocm.docs.amd.com/projects/hipFFT/en/docs-7.0.2/reference/fft-api-usage.html)
- [hipFFT Overview](https://rocm.docs.amd.com/projects/hipFFT/en/docs-7.1.0/conceptual/overview.html)
- [HIPFORT API Reference](https://rocm.docs.amd.com/projects/hipfort/en/latest/doxygen/html/md_input_supported_api_hipfft.html)

### 10.2 源代码

- [ROCm/hipFFT GitHub](https://github.com/ROCm/hipFFT) - ROCm ≤ 6.4.3
- [ROCm/rocm-libraries GitHub](https://github.com/ROCm/rocm-libraries) - ROCm ≥ 7.0.0

### 10.3 参考资料

- [hcFFT GitHub Archive](https://github.com/rocmarchive/hcFFT) - 历史版本参考
- [rocFFT Documentation](https://rocm.docs.amd.com/projects/rocFFT/en/latest/) - 后端实现

---

## 11. API 总览

hipFFT 对外暴露约 **30+ 个公开 API 函数**，涵盖以下类别：

| 类别 | 函数数量 | 主要功能 |
|------|---------|---------|
| 计划创建 | 4 | 1D/2D/3D/Many 维度 FFT 计划 |
| 执行函数 | 6 | 单/双精度，R2C/C2R/C2C 等变换 |
| 资源管理 | 4 | 创建/销毁/流管理 |
| 多 GPU 扩展 | 若干 | 跨 GPU FFT 执行 |

---

**文档版本**：基于 ROCm 7.1.0
**最后更新**：2026-03-12
