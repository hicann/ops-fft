# hipTensor API 接口总结

本文档总结了 hipTensor 仓库对外暴露的所有 API 接口。hipTensor 是 AMD ROCm 的张量运算库，类似于 NVIDIA 的 cuTensor。

---

## 1. 核心句柄管理 (Handle Management)

| 函数 | 说明 |
|------|------|
| `hiptensorCreate()` | 创建并初始化 hipTensor 库句柄 |
| `hiptensorDestroy()` | 销毁 hipTensor 库句柄并释放资源 |
| `hiptensorHandleResizePlanCache()` | 调整 Plan 缓存大小 |
| `hiptensorHandleWritePlanCacheToFile()` | 将 Plan 缓存写入文件 |
| `hiptensorHandleReadPlanCacheFromFile()` | 从文件读取 Plan 缓存 |
| `hiptensorWriteKernelCacheToFile()` | 将内核缓存写入文件 |
| `hiptensorReadKernelCacheFromFile()` | 从文件读取内核缓存 |

### 函数签名

```c
// 创建句柄
hiptensorStatus_t hiptensorCreate(hiptensorHandle_t* handle);

// 销毁句柄
hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t handle);

// 调整 Plan 缓存大小
hiptensorStatus_t hiptensorHandleResizePlanCache(hiptensorHandle_t handle,
                                                  const uint32_t numEntries);

// Plan 缓存文件操作
hiptensorStatus_t hiptensorHandleWritePlanCacheToFile(const hiptensorHandle_t handle,
                                                       const char fileName[]);
hiptensorStatus_t hiptensorHandleReadPlanCacheFromFile(hiptensorHandle_t handle,
                                                        const char fileName[],
                                                        uint32_t* numCachelinesRead);

// Kernel 缓存文件操作
hiptensorStatus_t hiptensorWriteKernelCacheToFile(const hiptensorHandle_t handle,
                                                   const char fileName[]);
hiptensorStatus_t hiptensorReadKernelCacheFromFile(hiptensorHandle_t handle,
                                                    const char fileName[]);
```

---

## 2. 张量描述符管理 (Tensor Descriptor)

| 函数 | 说明 |
|------|------|
| `hiptensorCreateTensorDescriptor()` | 创建张量描述符（维度、步长、数据类型） |
| `hiptensorDestroyTensorDescriptor()` | 销毁张量描述符 |

### 函数签名

```c
hiptensorStatus_t hiptensorCreateTensorDescriptor(
    const hiptensorHandle_t      handle,
    hiptensorTensorDescriptor_t* desc,
    const uint32_t               numModes,      // 维度数量
    const int64_t                lens[],        // 各维度长度
    const int64_t                strides[],     // 各维度步长（NULL 则使用列主序）
    hiptensorDataType_t          dataType,      // 数据类型
    uint32_t                     alignmentRequirement);  // 内存对齐要求

hiptensorStatus_t hiptensorDestroyTensorDescriptor(hiptensorTensorDescriptor_t desc);
```

### 参数说明

| 参数 | 方向 | 描述 |
|------|------|------|
| `handle` | [in] | hipTensor 库句柄 |
| `desc` | [out] | 张量描述符指针 |
| `numModes` | [in] | 张量维度（mode）数量 |
| `lens` | [in] | 各维度长度数组 |
| `strides` | [in] | 各维度步长数组，NULL 表示使用列主序布局 |
| `dataType` | [in] | 张量元素的数据类型 |
| `alignmentRequirement` | [in] | 内存对齐要求（字节） |

---

## 3. 操作描述符管理 (Operation Descriptor)

| 函数 | 说明 |
|------|------|
| `hiptensorCreateContraction()` | 创建张量收缩操作：D = α*A*B + β*C |
| `hiptensorCreatePermutation()` | 创建张量置换操作 |
| `hiptensorCreateReduction()` | 创建张量归约操作：D = α*opReduce(A) + β*C |
| `hiptensorCreateElementwiseBinary()` | 创建二元逐元素操作 |
| `hiptensorCreateElementwiseTrinary()` | 创建三元逐元素操作 |
| `hiptensorDestroyOperationDescriptor()` | 销毁操作描述符 |
| `hiptensorOperationDescriptorSetAttribute()` | 设置操作描述符属性 |
| `hiptensorOperationDescriptorGetAttribute()` | 获取操作描述符属性 |

### 3.1 张量收缩 (Contraction)

```c
hiptensorStatus_t hiptensorCreateContraction(
    const hiptensorHandle_t            handle,
    hiptensorOperationDescriptor_t*    desc,
    const hiptensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    hiptensorOperator_t                opA,
    const hiptensorTensorDescriptor_t  descB,
    const int32_t                      modeB[],
    hiptensorOperator_t                opB,
    const hiptensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    hiptensorOperator_t                opC,
    const hiptensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    const hiptensorComputeDescriptor_t descCompute);
```

**数学表达式**：D = α * A * B + β * C

| 参数 | 方向 | 描述 |
|------|------|------|
| `descA/B/C/D` | [in] | 张量 A/B/C/D 的描述符 |
| `modeA/B/C/D` | [in] | 张量 A/B/C/D 的模式（维度标签）数组 |
| `opA/B/C` | [in] | 应用于 A/B/C 的单目运算符 |
| `descCompute` | [in] | 中间计算的数据类型 |

### 3.2 张量置换 (Permutation)

```c
hiptensorStatus_t hiptensorCreatePermutation(
    const hiptensorHandle_t            handle,
    hiptensorOperationDescriptor_t*    desc,
    const hiptensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    hiptensorOperator_t                opA,
    const hiptensorTensorDescriptor_t  descB,
    const int32_t                      modeB[],
    const hiptensorComputeDescriptor_t descCompute);
```

**数学表达式**：B = α * opA(A)

### 3.3 张量归约 (Reduction)

```c
hiptensorStatus_t hiptensorCreateReduction(
    const hiptensorHandle_t            handle,
    hiptensorOperationDescriptor_t*    desc,
    const hiptensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    hiptensorOperator_t                opA,
    const hiptensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    hiptensorOperator_t                opC,
    const hiptensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    hiptensorOperator_t                opReduce,      // 归约操作符
    const hiptensorComputeDescriptor_t descCompute);
```

**数学表达式**：D = α * opReduce(opA(A)) + β * opC(C)

### 3.4 二元逐元素操作 (Elementwise Binary)

```c
hiptensorStatus_t hiptensorCreateElementwiseBinary(
    const hiptensorHandle_t            handle,
    hiptensorOperationDescriptor_t*    desc,
    const hiptensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    hiptensorOperator_t                opA,
    const hiptensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    hiptensorOperator_t                opC,
    const hiptensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    hiptensorOperator_t                opAC,          // 二元操作符
    const hiptensorComputeDescriptor_t descCompute);
```

**数学表达式**：D = opAC(α * opA(A), γ * opC(C))

### 3.5 三元逐元素操作 (Elementwise Trinary)

```c
hiptensorStatus_t hiptensorCreateElementwiseTrinary(
    const hiptensorHandle_t            handle,
    hiptensorOperationDescriptor_t*    desc,
    const hiptensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    hiptensorOperator_t                opA,
    const hiptensorTensorDescriptor_t  descB,
    const int32_t                      modeB[],
    hiptensorOperator_t                opB,
    const hiptensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    hiptensorOperator_t                opC,
    const hiptensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    hiptensorOperator_t                opAB,          // 第一个二元操作符
    hiptensorOperator_t                opABC,         // 第二个二元操作符
    const hiptensorComputeDescriptor_t descCompute);
```

**数学表达式**：D = opABC(opAB(α * opA(A), β * opB(B)), γ * opC(C))

### 3.6 属性操作

```c
// 设置属性
hiptensorStatus_t hiptensorOperationDescriptorSetAttribute(
    const hiptensorHandle_t                 handle,
    hiptensorOperationDescriptor_t          desc,
    hiptensorOperationDescriptorAttribute_t attr,
    const void*                             buf,
    size_t                                  sizeInBytes);

// 获取属性
hiptensorStatus_t hiptensorOperationDescriptorGetAttribute(
    const hiptensorHandle_t                 handle,
    hiptensorOperationDescriptor_t          desc,
    hiptensorOperationDescriptorAttribute_t attr,
    void*                                   buf,
    size_t                                  sizeInBytes);
```

**可用属性** (hiptensorOperationDescriptorAttribute_t):

| 属性 | 说明 |
|------|------|
| `HIPTENSOR_OPERATION_DESCRIPTOR_TAG` | 操作标签 |
| `HIPTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE` | 标量类型 |
| `HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS` | FLOPS 计数 |
| `HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES` | 数据移动字节数 |
| `HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_*` | 填充相关属性 |

---

## 4. Plan 和 PlanPreference 管理

| 函数 | 说明 |
|------|------|
| `hiptensorCreatePlanPreference()` | 创建 Plan 偏好设置 |
| `hiptensorDestroyPlanPreference()` | 销毁 Plan 偏好设置 |
| `hiptensorPlanPreferenceSetAttribute()` | 设置 Plan 偏好属性 |
| `hiptensorCreatePlan()` | 创建执行计划（选择合适的内核） |
| `hiptensorDestroyPlan()` | 销毁执行计划 |
| `hiptensorPlanGetAttribute()` | 获取 Plan 属性 |

### 4.1 PlanPreference

```c
hiptensorStatus_t hiptensorCreatePlanPreference(
    const hiptensorHandle_t    handle,
    hiptensorPlanPreference_t* pref,
    hiptensorAlgo_t            algo,      // 算法选择
    hiptensorJitMode_t         jitMode);  // JIT 模式

hiptensorStatus_t hiptensorDestroyPlanPreference(hiptensorPlanPreference_t pref);

hiptensorStatus_t hiptensorPlanPreferenceSetAttribute(
    const hiptensorHandle_t            handle,
    hiptensorPlanPreference_t          pref,
    hiptensorPlanPreferenceAttribute_t attr,
    const void*                        buf,
    size_t                             sizeInBytes);
```

**PlanPreference 属性** (hiptensorPlanPreferenceAttribute_t):

| 属性 | 说明 |
|------|------|
| `HIPTENSOR_PLAN_PREFERENCE_AUTOTUNE_MODE` | 自动调优模式 |
| `HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE` | 缓存模式 |
| `HIPTENSOR_PLAN_PREFERENCE_INCREMENTAL_COUNT` | 增量计数 |
| `HIPTENSOR_PLAN_PREFERENCE_ALGO` | 算法选择 |
| `HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK` | 内核秩 |
| `HIPTENSOR_PLAN_PREFERENCE_JIT` | JIT 模式 |

### 4.2 Plan

```c
hiptensorStatus_t hiptensorCreatePlan(
    const hiptensorHandle_t              handle,
    hiptensorPlan_t*                     plan,
    const hiptensorOperationDescriptor_t desc,
    const hiptensorPlanPreference_t      pref,              // 可为 NULL
    uint64_t                             workspaceSizeLimit);

hiptensorStatus_t hiptensorDestroyPlan(hiptensorPlan_t plan);

hiptensorStatus_t hiptensorPlanGetAttribute(
    const hiptensorHandle_t  handle,
    const hiptensorPlan_t    plan,
    hiptensorPlanAttribute_t attr,
    void*                    buf,
    size_t                   sizeInBytes);
```

**Plan 属性** (hiptensorPlanAttribute_t):

| 属性 | 说明 |
|------|------|
| `HIPTENSOR_PLAN_REQUIRED_WORKSPACE` | 所需工作空间大小 |

---

## 5. 核心计算执行函数 (Execution)

| 函数 | 说明 |
|------|------|
| `hiptensorContract()` | 执行张量收缩：D = α*A*B + β*C |
| `hiptensorPermute()` | 执行张量置换 |
| `hiptensorReduce()` | 执行张量归约 |
| `hiptensorElementwiseBinaryExecute()` | 执行二元逐元素操作 |
| `hiptensorElementwiseTrinaryExecute()` | 执行三元逐元素操作 |

### 5.1 张量收缩执行

```c
hiptensorStatus_t hiptensorContract(
    const hiptensorHandle_t handle,
    const hiptensorPlan_t   plan,
    const void*             alpha,        // A*B 的缩放因子
    const void*             A,            // 张量 A 数据
    const void*             B,            // 张量 B 数据
    const void*             beta,         // C 的缩放因子
    const void*             C,            // 张量 C 数据
    void*                   D,            // 输出张量 D
    void*                   workspace,    // 工作空间（可为 NULL）
    uint64_t                workspaceSize,
    hipStream_t             stream);
```

### 5.2 张量置换执行

```c
hiptensorStatus_t hiptensorPermute(
    const hiptensorHandle_t handle,
    const hiptensorPlan_t   plan,
    const void*             alpha,        // A 的缩放因子
    const void*             A,            // 输入张量
    void*                   B,            // 输出张量
    const hipStream_t       stream);
```

### 5.3 张量归约执行

```c
hiptensorStatus_t hiptensorReduce(
    const hiptensorHandle_t handle,
    const hiptensorPlan_t   plan,
    const void*             alpha,
    const void*             A,
    const void*             beta,
    const void*             C,
    void*                   D,
    void*                   workspace,
    uint64_t                workspaceSize,
    hipStream_t             stream);
```

### 5.4 二元逐元素执行

```c
hiptensorStatus_t hiptensorElementwiseBinaryExecute(
    const hiptensorHandle_t handle,
    const hiptensorPlan_t   plan,
    const void*             alpha,        // A 的缩放因子
    const void*             A,
    const void*             gamma,        // C 的缩放因子
    const void*             C,
    void*                   D,            // 输出
    hipStream_t             stream);
```

### 5.5 三元逐元素执行

```c
hiptensorStatus_t hiptensorElementwiseTrinaryExecute(
    const hiptensorHandle_t handle,
    const hiptensorPlan_t   plan,
    const void*             alpha,        // A 的缩放因子
    const void*             A,
    const void*             beta,         // B 的缩放因子
    const void*             B,
    const void*             gamma,        // C 的缩放因子
    const void*             C,
    void*                   D,            // 输出
    hipStream_t             stream);
```

---

## 6. 辅助工具函数

| 函数 | 说明 |
|------|------|
| `hiptensorEstimateWorkspaceSize()` | 估算工作空间大小 |
| `hiptensorGetErrorString()` | 获取错误码对应的描述字符串 |
| `hiptensorGetVersion()` | 获取 hipTensor 版本号 |
| `hiptensorGetHiprtVersion()` | 获取 HIP 运行时版本 |

### 函数签名

```c
hiptensorStatus_t hiptensorEstimateWorkspaceSize(
    const hiptensorHandle_t              handle,
    const hiptensorOperationDescriptor_t desc,
    const hiptensorPlanPreference_t      planPref,
    const hiptensorWorksizePreference_t  workspacePref,
    uint64_t*                            workspaceSizeEstimate);

const char* hiptensorGetErrorString(const hiptensorStatus_t error);

size_t hiptensorGetVersion();    // 返回: major*10000 + minor*100 + patch

int hiptensorGetHiprtVersion();
```

### 工作空间偏好

| 枚举值 | 说明 |
|--------|------|
| `HIPTENSOR_WORKSPACE_MIN` | 最小工作空间，至少有一个算法可用 |
| `HIPTENSOR_WORKSPACE_DEFAULT` | 默认工作空间，最合适的算法可用 |
| `HIPTENSOR_WORKSPACE_MAX` | 最大工作空间，所有算法都可用 |

---

## 7. 日志系统 (Logging)

| 函数 | 说明 |
|------|------|
| `hiptensorLoggerSetCallback()` | 设置日志回调函数 |
| `hiptensorLoggerSetFile()` | 设置日志输出文件流 |
| `hiptensorLoggerOpenFile()` | 打开日志文件 |
| `hiptensorLoggerSetLevel()` | 设置日志级别 |
| `hiptensorLoggerSetMask()` | 设置日志掩码 |
| `hiptensorLoggerForceDisable()` | 禁用日志 |

### 函数签名

```c
// 日志回调函数类型
typedef void (*hiptensorLoggerCallback_t)(int32_t     logContext,
                                           const char* funcName,
                                           const char* msg);

hiptensorStatus_t hiptensorLoggerSetCallback(hiptensorLoggerCallback_t callback);
hiptensorStatus_t hiptensorLoggerSetFile(FILE* file);
hiptensorStatus_t hiptensorLoggerOpenFile(const char* logFile);
hiptensorStatus_t hiptensorLoggerSetLevel(hiptensorLogLevel_t level);
hiptensorStatus_t hiptensorLoggerSetMask(int32_t mask);
hiptensorStatus_t hiptensorLoggerForceDisable();
```

### 日志级别

| 枚举值 | 说明 |
|--------|------|
| `HIPTENSOR_LOG_LEVEL_OFF` | 关闭日志 |
| `HIPTENSOR_LOG_LEVEL_ERROR` | 错误日志 |
| `HIPTENSOR_LOG_LEVEL_PERF_TRACE` | 性能跟踪日志 |
| `HIPTENSOR_LOG_LEVEL_PERF_HINT` | 性能提示日志 |
| `HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE` | 启发式选择日志 |
| `HIPTENSOR_LOG_LEVEL_API_TRACE` | API 调用跟踪日志 |

---

## 8. 数据类型 (Data Types)

### 8.1 基本数据类型枚举 (hiptensorDataType_t)

| 枚举值 | 说明 | 大小 |
|--------|------|------|
| `HIPTENSOR_R_32F` | 单精度浮点 | 4字节 |
| `HIPTENSOR_R_64F` | 双精度浮点 | 8字节 |
| `HIPTENSOR_R_16F` | 半精度浮点 | 2字节 |
| `HIPTENSOR_R_16BF` | BFloat16 | 2字节 |
| `HIPTENSOR_C_32F` | 单精度复数 | 8字节 |
| `HIPTENSOR_C_64F` | 双精度复数 | 16字节 |
| `HIPTENSOR_C_16F` | 半精度复数 | 4字节 |
| `HIPTENSOR_C_16BF` | BFloat16 复数 | 4字节 |
| `HIPTENSOR_R_8I` | 8位有符号整数 | 1字节 |
| `HIPTENSOR_R_8U` | 8位无符号整数 | 1字节 |
| `HIPTENSOR_R_32I` | 32位有符号整数 | 4字节 |
| `HIPTENSOR_R_32U` | 32位无符号整数 | 4字节 |
| `HIPTENSOR_R_64I` | 64位有符号整数 | 8字节 |
| `HIPTENSOR_R_64U` | 64位无符号整数 | 8字节 |
| `HIPTENSOR_R_4I/U` | 4位整数 | - |

### 8.2 计算类型枚举 (hiptensorComputeDescriptor_t)

| 枚举值 | 说明 |
|--------|------|
| `HIPTENSOR_COMPUTE_DESC_16F` | 半精度计算 |
| `HIPTENSOR_COMPUTE_DESC_16BF` | BFloat16 计算 |
| `HIPTENSOR_COMPUTE_DESC_32F` | 单精度计算 |
| `HIPTENSOR_COMPUTE_DESC_64F` | 双精度计算 |
| `HIPTENSOR_COMPUTE_DESC_C32F` | 单精度复数计算 |
| `HIPTENSOR_COMPUTE_DESC_C64F` | 双精度复数计算 |
| `HIPTENSOR_COMPUTE_DESC_NONE` | 无类型 |

### 8.3 不透明句柄类型

| 类型 | 说明 |
|------|------|
| `hiptensorHandle_t` | hipTensor 库上下文句柄 |
| `hiptensorTensorDescriptor_t` | 张量描述符 |
| `hiptensorOperationDescriptor_t` | 操作描述符 |
| `hiptensorPlanPreference_t` | Plan 偏好设置 |
| `hiptensorPlan_t` | 执行计划 |

### 8.4 hiptensorHandle_t 结构详解

`hiptensorHandle_t` 是 hipTensor 库的核心上下文句柄，存储库的运行时状态。

#### 8.4.1 结构体定义

```cpp
struct hiptensorHandle
{
    // 获取设备信息
    hiptensor::HipDevice getDevice() { return mDevice; }

    // 获取/设置 Plan 缓存
    hiptensor::PlanCache* getPlanCache() { return planCache; }
    void setPlanCache(hiptensor::PlanCache* pt_PlanCache) { planCache = pt_PlanCache; }

private:
    hiptensor::HipDevice  mDevice;              // 设备信息
    hiptensor::PlanCache* planCache = nullptr;  // Plan 缓存指针
};
```

#### 8.4.2 HipDevice 类

存储 HIP 设备的硬件信息和能力检测：

| 成员 | 类型 | 说明 |
|------|------|------|
| `mDeviceId` | `hipDevice_t` | HIP 设备 ID |
| `mProps` | `hipDeviceProp_t` | HIP 设备属性 |
| `mArch` | `hipDeviceArch_t` | 设备架构信息 |
| `mGcnArch` | `hipGcnArch_t` | GCN 架构枚举 |
| `mWarpSize` | `int` | Warp 大小 (32 或 64) |
| `mSharedMemSize` | `int` | 共享内存大小 |
| `mCuCount` | `int` | 计算单元数量 |
| `mMaxFreqMhz` | `int` | 最大频率 (MHz) |

**支持的 GCN 架构**：

| 枚举值 | 架构代码 | 对应产品 |
|--------|----------|----------|
| `GFX908` | 0x908 | MI100 |
| `GFX90A` | 0x90A | MI200 series |
| `GFX942` | 0x942 | MI300 series |
| `GFX950` | 0x950 | - |
| `GFX1100~1103` | 0x1100-1103 | RDNA3 series |
| `GFX1150~1153` | 0x1150-1153 | RDNA3.5 series |
| `GFX1200~1201` | 0x1200-1201 | Future |
| `UNSUPPORTED_ARCH` | 0x0 | 不支持的架构 |

**HipDevice 主要方法**：

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `getDeviceId()` | `hipDevice_t` | 获取设备 ID |
| `getGcnArch()` | `hipGcnArch_t` | 获取 GCN 架构 |
| `warpSize()` | `int` | 获取 Warp 大小 |
| `supportsF64()` | `bool` | 是否支持双精度 |
| `matrixCoreSupport(type)` | `bool` | 是否支持矩阵核心 |

#### 8.4.3 PlanCache 类

存储和管理已编译内核的缓存，避免重复编译：

| 成员 | 类型 | 说明 |
|------|------|------|
| `mMaxCachelines` | `uint32_t` | 缓存最大条目数 |
| `mPlanCacheLines` | `unordered_map<HashId, Uid>` | 哈希表：操作描述符哈希 → 解决方案 UID |
| `mPqUidUsedTimes` | `UpdatablePriorityQueue` | LRU 优先队列，按使用时间排序 |
| `mMutex` | `std::mutex` | 线程安全互斥锁 |

**PlanCache 主要方法**：

| 方法 | 说明 |
|------|------|
| `getHashID(desc)` | 根据操作描述符计算哈希 ID |
| `querySolutionUid(desc)` | 查询缓存的解决方案 UID |
| `addCacheLine(hashId, sol_id)` | 添加缓存条目（超出容量时淘汰 LRU） |
| `writeFile(fileName)` | 序列化到文件 |
| `readFile(fileName)` | 从文件反序列化 |
| `resize(numEntries)` | 调整缓存大小 |
| `getCachelinesNum()` | 获取当前缓存条目数 |

**哈希计算因子**：
```
{Tag, DatatypeA, DatatypeB, DatatypeC, DatatypeD, DatatypeCompute,
 mOperationType, mContractionOpId, lengths, strides, modes}
```

**UpdatablePriorityQueue 辅助类**：

用于实现 LRU 淘汰策略的可更新优先队列：

```cpp
template <typename T, typename PriorityType, typename Compare = std::less<PriorityType>>
class UpdatablePriorityQueue
{
    struct Element {
        T            key;           // 缓存键 (HashId)
        PriorityType priority;     // 优先级 (时间戳)
        std::size_t  heapIndex;    // 在堆中的索引，支持 O(1) 查找
    };

private:
    std::vector<Element*>           mHeapData;      // 堆数据
    std::unordered_map<T, Element*> mKeyToElement;  // 键到元素的映射
    Compare                         compare;        // 比较函数

    void siftUp(std::size_t index);   // 上浮调整
    void siftDown(std::size_t index); // 下沉调整

public:
    void push(const T& key, const PriorityType& priority);  // O(log n)
    bool updateItem(const T& key, const PriorityType& new_priority);  // O(log n)
    std::pair<T, PriorityType> top() const;  // O(1)
    void pop();  // O(log n)
    bool empty() const;
    std::size_t size() const;
};
```

**内部工作原理**：

1. **查询缓存流程**：
```
querySolutionUid(desc)
         │
         ▼
getHashID(desc)  ──────► 计算操作描述符哈希值
         │
         ▼
mPlanCacheLines.find(hashId)  ──► 在哈希表中查找
         │
         ├── 找到 ──► updateCachelineTime(hashId) ──► 更新 LRU 时间戳
         │                                    └─► 返回 Uid
         │
         └── 未找到 ──► 返回 0 (表示缓存未命中)
```

2. **添加缓存流程**：
```
addCacheLine(hashId, sol_id)
         │
         ▼
检查 mPlanCacheLines.size() >= mMaxCachelines ?
         │
         ├── 是 (缓存已满) ──►
         │                      mPqUidUsedTimes.top() ──► 获取最久未使用的 hashId
         │                              │
         │                              ▼
         │                      mPqUidUsedTimes.pop() ──► 从优先队列移除
         │                              │
         │                              ▼
         │                      mPlanCacheLines.erase(lru_hashId) ──► 从哈希表移除
         │
         └── 否 (缓存未满) ──► 直接添加
         │
         ▼
mPlanCacheLines[hashId] = sol_id  ──► 添加到哈希表
         │
         ▼
mPqUidUsedTimes.push(hashId, now) ──► 添加到优先队列（当前时间戳）
         │
         ▼
释放锁，完成
```

3. **LRU 淘汰机制详解**：
```
┌────────────────────────────────────────────────────────────────┐
│                    LRU 淘汰机制                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  时间轴:  t1 < t2 < t3 < t4 < t5                               │
│                                                                │
│  优先队列 (最小堆，按时间排序):                                   │
│                                                                │
│       [t1]              [t2]         [t3]         [t4]         │
│         │                / \           / \         / \          │
│         ▼               ▼   ▼         ▼   ▼       ▼   ▼         │
│       root            left right    left right   left right    │
│                                                                │
│  访问时更新:                                                    │
│  - 查询命中 → updateItem(hashId, currentTime)                  │
│  - 优先级更新为当前时间 → 元素移动到堆底                         │
│  - 堆顶始终是最久未使用的条目                                    │
│                                                                │
│  淘汰时:                                                        │
│  - top() 返回堆顶 (最旧)                                        │
│  - pop() 移除堆顶                                              │
│  - 从哈希表删除对应条目                                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

4. **序列化/反序列化**：
   - `writeFile()`: 遍历 mPlanCacheLines，将 HashId 和 Uid 写入二进制文件
   - `readFile()`: 从文件读取，重建哈希表和优先队列

5. **线程安全设计**：
```cpp
Uid querySolutionUid(hiptensorOperationDescriptor_t desc)
{
    std::lock_guard<std::mutex> lock(mMutex);  // RAII 锁
    // ... 查询逻辑
}

void addCacheLine(HashId hashId, Uid sol_id)
{
    std::lock_guard<std::mutex> lock(mMutex);  // RAII 锁
    // ... 添加逻辑
}
```

**性能特点**：

| 操作 | 数据结构 | 复杂度 | 说明 |
|------|----------|--------|------|
| 查询 | 哈希表 | O(1) | 平均情况 |
| 添加 | 哈希表 + 堆 | O(log n) | 堆插入 |
| 更新时间 | 堆 | O(log n) | 堆调整 |
| 淘汰 LRU | 堆 | O(log n) | 堆删除 |
| 序列化 | 遍历 | O(n) | 线性扫描 |

**解决方案 (Solution) 的本质**：

PlanCache 缓存的是**内核选择决策**，而不是编译后的内核代码：

```
┌─────────────────────────────────────────────────────────────────┐
│                    hipTensor 内核架构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           CK (Composable Kernels) 库                     │   │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │   │Kernel 1 │ │Kernel 2 │ │Kernel 3 │ │  ...    │       │   │
│  │   │Uid=1001 │ │Uid=1002 │ │Uid=1003 │ │Uid=xxxx │       │   │
│  │   └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  │         ▲                                            │   │
│  │         │ 预编译的内核模板（非 JIT）                    │   │
│  └─────────┼────────────────────────────────────────────┘   │
│            │                                                    │
│  ┌─────────▼─────────────────────────────────────────────┐   │
│  │              候选解决方案列表 (Candidates)              │   │
│  │   vector<ContractionSolution*>                         │   │
│  │   - 每个 Solution 封装一个 CK 内核指针                  │   │
│  │   - 每个 Solution 有唯一的 Uid                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**ContractionSolution 类结构**：
```cpp
class ContractionSolution
{
protected:
    std::unique_ptr<BaseOperator> mDeviceOp;      // CK 内核操作符
    std::unique_ptr<SolutionParams> mParams;      // 内核参数
    std::unique_ptr<BaseArgument> mInvokerArgPtr; // 调用参数
    std::unique_ptr<BaseInvoker> mInvokerPtr;     // 内核调用器

public:
    size_t uid() const;        // 唯一标识符 - PlanCache 缓存的值
    bool isValid() const;      // 是否适用于当前问题
    size_t workspaceSize();    // 所需工作空间
};
```

**内核选择流程对比**：

```
无缓存时:                          有缓存时:
─────────────────────────         ─────────────────────────
enumerateSolutions()              Uid = querySolutionUid(desc)
        │                                 │
        ▼                                 ▼
遍历所有候选内核 O(n) ────────►   findSolutionByUid() O(1)
        │                                 │
        ▼                                 ▼
每个候选检查 isValid()            直接使用缓存的内核
        │
        ▼
可能执行 Autotune（耗时！）
        │
        ▼
选择最优内核 winner
        │
        ▼
addCacheLine(hashId, winner->uid())
```

**PlanCache 缓存的真正价值**：

| 缓存内容 | 说明 |
|---------|------|
| **HashId** | 操作描述符的哈希值（唯一标识操作类型和参数组合） |
| **Uid** | 最优内核的唯一标识符 |

| 避免的开销 | 无缓存 | 有缓存 |
|-----------|--------|--------|
| 遍历候选内核 | O(n) 次检查 | O(1) 查找 |
| 有效性检查 | 每个候选都检查 | 跳过 |
| Autotune 性能测试 | 可能执行（耗时） | 跳过 |

**关键代码**：
```cpp
// hiptensor_contraction.cpp - 缓存命中时直接使用
if(handle->getPlanCache() && pref->mCacheMode == HIPTENSOR_CACHE_MODE_PEDANTIC)
{
    auto Uid = handle->getPlanCache()->querySolutionUid(desc);
    if(Uid > 0ull)
    {
        // 直接通过 Uid 找到最优内核，无需遍历
        winner = findSolutionByUid(candidates, Uid);
    }
}
```

#### 8.4.4 句柄生命周期

**hiptensorCreate 源码流程**：

```cpp
hiptensorStatus_t hiptensorCreate(hiptensorHandle_t* handle)
{
    (*handle) = new hiptensorHandle;  // ← HipDevice 构造函数隐式调用

    auto hip_status = hipInit(0);    // 初始化 HIP

    const char* plan_cache_disable = std::getenv("HIPTENSOR_DISABLE_PLAN_CACHE");
    if(plan_cache_disable == nullptr || strcmp(plan_cache_disable, "ON") != 0)
    {
        hiptensor::PlanCache* planCache = new hiptensor::PlanCache;
        (*handle)->setPlanCache(planCache);  // ← 显式创建 PlanCache
    }

    return HIPTENSOR_STATUS_SUCCESS;
}
```

**关键点：HipDevice 构造是隐式的**

```cpp
struct hiptensorHandle
{
private:
    hiptensor::HipDevice  mDevice;              // 成员变量（非指针）
    hiptensor::PlanCache* planCache = nullptr;
};
```

`mDevice` 是成员变量而非指针，当 `new hiptensorHandle` 执行时，C++ 会**自动调用** `HipDevice` 的默认构造函数。

**生命周期流程图**：

```
┌─────────────────────────────────────────────────────────────────┐
│                    hiptensorHandle 生命周期                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  hiptensorCreate()                                              │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────┐                       │
│  │ new hiptensorHandle                  │                       │
│  │                                      │                       │
│  │  内部自动执行:                        │                       │
│  │  - HipDevice() 构造函数（隐式）       │ ← C++ 自动调用        │
│  │    - hipGetDevice(&mDeviceId)        │                       │
│  │    - hipGetDeviceProperties(&mProps) │                       │
│  │    - 解析 gcnArch 字符串             │                       │
│  │    - 设置 warpSize/cuCount/freq      │                       │
│  │  - planCache = nullptr               │                       │
│  └──────────────────┬───────────────────┘                       │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────┐                       │
│  │ hipInit(0)                           │                       │
│  └──────────────────┬───────────────────┘                       │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────┐                       │
│  │ 检查 HIPTENSOR_DISABLE_PLAN_CACHE    │                       │
│  └──────────────────┬───────────────────┘                       │
│                     │                                           │
│          ┌─────────┴─────────┐                                  │
│          │                   │                                  │
│    未设置/非ON              ON                                  │
│          │                   │                                  │
│          ▼                   ▼                                  │
│  ┌──────────────────┐  (跳过PlanCache)                         │
│  │ new PlanCache   │                                           │
│  │ setPlanCache()   │                                           │
│  └──────────────────┘                                           │
│                     │                                           │
│                     ▼                                           │
│              [使用阶段]                                          │
│                     │                                           │
│                     ▼                                           │
│  hiptensorDestroy()                                             │
│       │                                                         │
│       ▼                                                         │
│  delete handle  ──► HipDevice::~HipDevice()（隐式析构）         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 8.4.5 设计特点

1. **轻量级设计**：Handle 本身只包含 2 个成员，复杂逻辑封装在 HipDevice 和 PlanCache 类中

2. **缓存优化**：
   - 使用 LRU 策略自动淘汰旧 Plan
   - 支持缓存持久化到文件
   - 可通过环境变量 `HIPTENSOR_DISABLE_PLAN_CACHE=ON` 禁用

3. **设备能力检测**：在创建时自动检测设备架构和能力，后续操作可快速查询

4. **线程安全**：PlanCache 使用 mutex 保护，支持多线程访问

#### 8.4.6 使用建议

```cpp
// 创建句柄（自动检测设备、启用缓存）
hiptensorHandle_t handle;
hiptensorCreate(&handle);

// 调整缓存大小（可选）
hiptensorHandleResizePlanCache(handle, 1000);

// 保存缓存供下次使用（可选）
hiptensorHandleWritePlanCacheToFile(handle, "plan_cache.bin");

// 下次启动时加载缓存
uint32_t numRead;
hiptensorHandleReadPlanCacheFromFile(handle, "plan_cache.bin", &numRead);

// 使用完后销毁
hiptensorDestroy(handle);
```

### 8.5 hiptensorTensorDescriptor_t 结构详解

`hiptensorTensorDescriptor_t` 是张量的元数据描述符，用于描述张量的形状、布局和数据类型，而不包含实际的数据内容。

#### 8.5.1 概念说明

**张量描述符 vs 张量数据**：

| 概念 | 比喻 | 内容 |
|------|------|------|
| 张量描述符 | 房屋蓝图 | 维度、步长、数据类型等**元信息** |
| 张量数据 | 实际房屋 | 内存中存储的**实际数值** |

```
┌──────────────────────────────────────────────────────────────┐
│                   张量描述符 (Descriptor)                     │
├──────────────────────────────────────────────────────────────┤
│  mType = HIPTENSOR_R_32F                                     │
│  mLengths = [1024, 1024]      // 形状: 1024x1024             │
│  mStrides = [1024, 1]         // 内存布局                     │
│  mAlignmentRequirement = 4    // 对齐要求                     │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ 描述
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   张量数据 (Device Memory)                    │
├──────────────────────────────────────────────────────────────┤
│  地址 0x7f000000: [1.0, 2.0, 3.0, 4.0, ...]                  │
│  地址 0x7f001000: [5.0, 6.0, 7.0, 8.0, ...]                  │
│  ...                                                         │
└──────────────────────────────────────────────────────────────┘
```

#### 8.5.2 结构体定义

```cpp
struct hiptensorTensorDescriptor
{
    hiptensorDataType_t         mType;                   // 数据类型
    std::vector<std::size_t>    mLengths;                // 各维度长度
    std::vector<std::size_t>    mStrides;                // 各维度步长
    uint32_t                    mAlignmentRequirement;   // 内存对齐要求
};
```

#### 8.5.3 成员详解

| 成员 | 类型 | 说明 |
|------|------|------|
| `mType` | `hiptensorDataType_t` | 张量元素的数据类型（如 `HIPTENSOR_R_32F`） |
| `mLengths` | `vector<size_t>` | 各维度的长度数组，例如 `[M, N, K]` |
| `mStrides` | `vector<size_t>` | 各维度的步长数组，描述内存布局 |
| `mAlignmentRequirement` | `uint32_t` | 内存对齐要求（字节），通常为 4/8/16 |

#### 8.5.4 Stride（步长）概念详解

**什么是步长？**

步长表示在某个维度上移动一个位置时，需要在内存中跳过多少个元素。

```
3D 张量示例: A[2][3][4]

内存布局（行主序）:
索引计算: index = i * stride[0] + j * stride[1] + k * stride[2]

        ┌─────────────────────────────────────┐
        │      k=0    k=1    k=2    k=3       │
        ├─────────────────────────────────────┤
i=0,j=0 │  [0]    [1]    [2]    [3]          │
i=0,j=1 │  [4]    [5]    [6]    [7]          │
i=0,j=2 │  [8]    [9]    [10]   [11]         │
        ├─────────────────────────────────────┤
i=1,j=0 │  [12]   [13]   [14]   [15]         │
i=1,j=1 │  [16]   [17]   [18]   [19]         │
i=1,j=2 │  [20]   [21]   [22]   [23]         │
        └─────────────────────────────────────┘

行主序步长: strides = [12, 4, 1]
- stride[0]=12: 移动一个 i 需要 12 个元素
- stride[1]=4:  移动一个 j 需要 4 个元素
- stride[2]=1:  移动一个 k 需要 1 个元素
```

**步长计算公式**：

```
行主序（C-style）:
stride[n-1] = 1
stride[i] = stride[i+1] * length[i+1]   (从后向前计算)

列主序（Fortran-style）:
stride[0] = 1
stride[i] = stride[i-1] * length[i-1]   (从前向后计算)
```

#### 8.5.5 行主序 vs 列主序

| 特性 | 行主序 (Row-major) | 列主序 (Column-major) |
|------|-------------------|---------------------|
| 也称为 | C-style | Fortran-style |
| 最后维度 | 连续存储 | 非连续存储 |
| 第一维度 | 非连续存储 | 连续存储 |
| 语言 | C/C++, Python (NumPy 默认) | Fortran, MATLAB |
| A[i][j] 相邻 | j 连续 | i 连续 |

**2D 矩阵示例**：

```
矩阵: A[2][3]

行主序存储 (strides = [3, 1]):
内存: [A[0][0], A[0][1], A[0][2], A[1][0], A[1][1], A[1][2]]
       └─ row 0 ─┘              └─ row 1 ─┘

列主序存储 (strides = [1, 2]):
内存: [A[0][0], A[1][0], A[0][1], A[1][1], A[0][2], A[1][2]]
       └─ col 0 ─┘  └─ col 1 ─┘  └─ col 2 ─┘
```

#### 8.5.6 自动步长计算

当调用 `hiptensorCreateTensorDescriptor` 时，如果传入 `strides = NULL`，库会自动计算默认步长：

```cpp
// 源码位置: hiptensor.cpp
hiptensorStatus_t hiptensorCreateTensorDescriptor(
    const hiptensorHandle_t      handle,
    hiptensorTensorDescriptor_t* desc,
    const uint32_t               numModes,
    const int64_t                lens[],
    const int64_t                strides[],      // NULL 表示自动计算
    hiptensorDataType_t          dataType,
    uint32_t                     alignmentRequirement)
{
    *desc = new hiptensorTensorDescriptor();

    // 复制维度
    (*desc)->mLengths.assign(lens, lens + numModes);

    if(strides == nullptr)
    {
        // 自动计算步长（列主序）
        (*desc)->mStrides.resize(numModes);
        (*desc)->mStrides[0] = 1;
        for(uint32_t i = 1; i < numModes; i++)
        {
            (*desc)->mStrides[i] = (*desc)->mStrides[i-1] * lens[i-1];
        }
    }
    else
    {
        // 使用用户提供的步长
        (*desc)->mStrides.assign(strides, strides + numModes);
    }

    (*desc)->mType = dataType;
    (*desc)->mAlignmentRequirement = alignmentRequirement;

    return HIPTENSOR_STATUS_SUCCESS;
}
```

**注意**：hipTensor 默认使用**列主序**计算步长（与 cuFFT 不同）。

#### 8.5.7 在操作描述符中的作用

张量描述符在创建操作描述符时被引用，用于描述操作的输入/输出张量：

```cpp
// 张量收缩: D = α*A*B + β*C
hiptensorCreateContraction(
    handle, &opDesc,
    descA, modeA, opA,    // A 张量的描述符、模式、操作符
    descB, modeB, opB,    // B 张量
    descC, modeC, opC,    // C 张量
    descD, modeD,         // D 张量（输出）
    HIPTENSOR_COMPUTE_DESC_32F);
```

**Mode（模式）的作用**：

Mode 数组定义了张量各维度在运算中的角色（标签）：

```
矩阵乘法: C(m,n) = A(m,k) * B(k,n)

modeA = [0, 2]  // 维度 0=m, 维度 1=k
modeB = [2, 1]  // 维度 0=k, 维度 1=n
modeC = [0, 1]  // 维度 0=m, 维度 1=n

相同的 mode 值表示需要收缩（求和）的维度
         不同的 mode 值表示保留的维度
```

#### 8.5.8 使用示例

**示例 1: 创建行主序张量描述符**

```cpp
// 2D 张量: 1024 x 512, 行主序
int64_t extent[] = {1024, 512};
int64_t strides[] = {512, 1};  // 显式指定行主序

hiptensorTensorDescriptor_t desc;
hiptensorCreateTensorDescriptor(
    handle, &desc,
    2,            // numModes
    extent,       // lens
    strides,      // 显式步长（行主序）
    HIPTENSOR_R_32F,
    4);           // 4字节对齐
```

**示例 2: 创建非连续内存布局的张量**

```cpp
// 子矩阵: 从大矩阵中提取一部分
// 原矩阵: 2048 x 2048, 只使用前 512 x 512
int64_t extent[] = {512, 512};
int64_t strides[] = {2048, 1};  // 原矩阵的步长

hiptensorTensorDescriptor_t desc;
hiptensorCreateTensorDescriptor(
    handle, &desc,
    2, extent, strides,
    HIPTENSOR_R_32F, 4);

// 后续操作时传入子矩阵的起始指针即可
```

**示例 3: 使用默认步长**

```cpp
// 让库自动计算步长（列主序）
int64_t extent[] = {256, 256, 128};

hiptensorTensorDescriptor_t desc;
hiptensorCreateTensorDescriptor(
    handle, &desc,
    3, extent,
    NULL,        // NULL = 自动计算步长
    HIPTENSOR_R_32F, 4);

// 自动计算的步长: [1, 256, 65536]
```

#### 8.5.9 为什么需要张量描述符？

| 原因 | 说明 |
|------|------|
| **灵活性** | 支持任意内存布局（行主序、列主序、非连续） |
| **性能优化** | 库可以根据布局选择最优内核 |
| **类型安全** | 编译时检查数据类型一致性 |
| **内存效率** | 避免数据拷贝，直接在原始布局上操作 |

**无描述符时的困境**：

```
如果没有描述符:
- 库不知道数据的内存布局
- 无法处理非连续内存
- 必须约定固定的布局（如总是行主序）
- 每次调用都要传递大量参数
```

**有描述符后的优势**：

```
使用描述符:
- 一次创建，多次使用
- 清晰地表达数据语义
- 支持复杂的内存布局
- 便于内核优化
```

#### 8.5.10 为什么创建描述符需要 handle

在 `hiptensorCreateTensorDescriptor` 接口中，handle 参数起到两个关键作用：

**1. 验证库已初始化**

```cpp
if(handle == nullptr)
{
    return HIPTENSOR_STATUS_NOT_INITIALIZED;
}
```

**2. 设备能力检测（核心作用）**

```cpp
if(dataType == HIPTENSOR_R_64F && !handle->getDevice().supportsF64())
{
    return HIPTENSOR_STATUS_ARCH_MISMATCH;
}
```

并非所有 AMD GPU 都支持所有数据类型：

| 架构 | 产品 | 双精度 (F64) | 矩阵核心 (F32) |
|------|------|-------------|---------------|
| GFX908 | MI100 | ❌ | ✅ |
| GFX90A | MI200 | ✅ | ✅ |
| GFX942 | MI300 | ✅ | ✅ |
| GFX1100+ | RDNA3 | ❌ | ❌ |

**设计原则**：提前失败（fail-fast）—— 在创建阶段检测不支持的配置，而非执行时才报错。

### 8.6 hiptensorOperationDescriptor_t 结构详解

操作描述符是完整的"计算配方"，描述了张量运算的所有细节。

#### 8.6.1 结构体定义

```cpp
struct hiptensorOperationDescriptor
{
    // 基本属性
    uint32_t            mTag;           // 用户标签
    hiptensorDataType_t mScalarType;    // 标量类型（α/β）
    float               mFlops;         // 计算量估算
    float               mMovedBytes;    // 数据移动量
    uint32_t            mPaddingLeft;   // 左填充
    uint32_t            mPaddingRighT;  // 右填充
    void*               mPaddingValue;  // 填充值

    // 操作类型
    hiptensorOperationType_t mOperationType;    // 操作大类
    int32_t                  mContractionOpId;  // 收缩子类型

    // 张量 A/B/C/D 及其模式、操作符
    hiptensorTensorDescriptor_t mDescA, mDescB, mDescC, mDescD;
    std::vector<int32_t>        mModeA, mModeB, mModeC, mModeD;
    hiptensorOperator_t         mOpA, mOpB, mOpC;

    // 计算精度
    hiptensorComputeDescriptor_t mDescCompute;
};
```

#### 8.6.2 mOperationType（操作大类）

决定**做什么类型的操作**：

| 枚举值 | 说明 | 数学形式 |
|--------|------|---------|
| `HIPTENSOR_CONTRACTION` | 张量收缩 | D = α*A*B + β*C |
| `HIPTENSOR_REDUCTION` | 张量归约 | D = α*reduce(A) + β*C |
| `HIPTENSOR_PERMUTATION` | 张量置换 | B = α*A |
| `HIPTENSOR_ELEMENTWISE_BINARY` | 二元逐元素 | D = opAC(α*A, γ*C) |
| `HIPTENSOR_ELEMENTWISE_TRINARY` | 三元逐元素 | D = opABC(opAB(α*A,β*B), γ*C) |

#### 8.6.3 mContractionOpId（收缩子类型）

仅当 `mOperationType = HIPTENSOR_CONTRACTION` 时有效，进一步细分收缩操作：

| 枚举值 | 数学形式 | 说明 |
|--------|---------|------|
| `SCALE` | D = α*A*B | 无 C，实数 |
| `BILINEAR` | D = α*A*B + β*C | 有 C，实数 |
| `SCALE_COMPLEX` | D = α*A*B | 无 C，复数 |
| `BILINEAR_COMPLEX` | D = α*A*B + β*C | 有 C，复数 |
| `BILINEAR_UNARY` | D = α*op(A)*op(B) + β*op(C) | 有单目操作符 |

#### 8.6.4 为什么收缩"只有加法"

这是**数学定义**决定的。张量收缩本质是 **Einstein 求和约定**：

```
C[i,j] = Σ_k A[i,k] * B[k,j]
         ↑
       乘法+求和是收缩的数学定义，不可更改
```

公式解析：

```
D = α * A * B + β * C
    └───┬───┘   └──┬──┘
        │          │
    收缩运算     偏置累加
   (乘法+求和)   (固定加法)
```

**实现其他操作的方式**：

| 操作 | 实现方式 |
|------|---------|
| D = A*B - C | 收缩，设置 β = -1 |
| D = A + C | ElementwiseBinary + OP_ADD |
| D = A * C | ElementwiseBinary + OP_MUL |
| D = max(A, C) | ElementwiseBinary + OP_MAX |

#### 8.6.5 α/β 赋值时机

α 和 β 在**执行阶段**传入，而非创建描述符时：

```cpp
// 创建描述符 - 不涉及 α/β
hiptensorCreateContraction(handle, &opDesc, ...);

// 创建 Plan - 也不涉及 α/β
hiptensorCreatePlan(handle, &plan, opDesc, pref, workspaceSize);

// 执行时才传入 α/β
float alpha = 1.0f, beta = 0.0f;
hiptensorContract(handle, plan, &alpha, A, B, &beta, C, D, ...);
```

**设计原因**：一次创建 Plan，可多次执行（不同 α/β/数据）。

#### 8.6.6 基本属性详解

操作描述符包含一组可选的基本属性，主要用于**用户标记**和**性能分析**：

| 属性 | 类型 | 默认值 | 用途 | 参与缓存哈希 |
|------|------|--------|------|-------------|
| `mTag` | uint32 | 0 | 用户自定义标记 | ✅ 是 |
| `mScalarType` | enum | 自动 | α/β 标量类型（从 descCompute 转换） | ❌ 否 |
| `mFlops` | float | 0 | 计算量估算（用户设置） | ❌ 否 |
| `mMovedBytes` | float | 0 | 数据移动量估算（用户设置） | ❌ 否 |
| `mPaddingLeft` | uint32 | 0 | 左填充元素数 | ❌ 否 |
| `mPaddingRighT` | uint32 | 0 | 右填充元素数 | ❌ 否 |
| `mPaddingValue` | void* | nullptr | 填充值 | ❌ 否 |

**各属性详解**：

**1. mTag（用户标签）**

唯一参与缓存哈希计算的基本属性，用于区分相同形状但不同用途的操作：

```cpp
// 设置 Tag
uint32_t tag = 12345;
hiptensorOperationDescriptorSetAttribute(handle, opDesc,
    HIPTENSOR_OPERATION_DESCRIPTOR_TAG, &tag, sizeof(tag));
```

**2. mFlops / mMovedBytes（性能分析）**

用户可手动设置，用于计算 Arithmetic Intensity (AI = FLOPS / Bytes)：

```cpp
// 矩阵乘法: D = A * B, A(M,K), B(K,N)
float flops = 2.0f * M * N * K;  // 每个元素 K 次乘法 + K 次加法
float movedBytes = (M*K + K*N + M*N) * sizeof(float);

hiptensorOperationDescriptorSetAttribute(handle, opDesc,
    HIPTENSOR_OPERATION_DESCRIPTOR_FLOPS, &flops, sizeof(flops));
hiptensorOperationDescriptorSetAttribute(handle, opDesc,
    HIPTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES, &movedBytes, sizeof(movedBytes));
```

**3. Padding 相关属性**

用于描述张量边界的填充情况（**注意：当前版本未实际使用，为预留接口**）：

```cpp
// 示例：3x3 矩阵，最内层维度填充到 4
// 1 2 3 [x]
// 4 5 6 [x]
// 7 8 9 [x]

uint32_t paddingRight = 1;  // 每行末尾填充1个元素
float paddingValue = 0.0f;  // 填充值为0

hiptensorOperationDescriptorSetAttribute(handle, opDesc,
    HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT, &paddingRight, sizeof(paddingRight));
hiptensorOperationDescriptorSetAttribute(handle, opDesc,
    HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE, &paddingValue, sizeof(paddingValue));
```

**Padding 应用场景推测**：

| 场景 | Padding 方向 | 用途 |
|------|-------------|------|
| 分块计算 | Right | 最后一块不足时填充对齐 |
| RNN 序列 | Right | 对齐不同长度序列 |
| 信号处理 | Left/Right | 卷积边界处理 |
| Transformer | Left/Right | Attention mask |

> ⚠️ **注意**：Padding 属性在 hipTensor 当前版本中**未实际使用**，仅作为与 cuTensor API 兼容的预留接口。实际 padding 通常由上层框架（PyTorch/TensorFlow）处理。

### 8.7 hiptensorPlanPreference_t 结构详解

**PlanPreference** 是"内核选择偏好设置"，用于指导 `hiptensorCreatePlan` 如何从多个候选内核中选择最优的一个。

> **注意**：PlanPreference 是**可选的**，`hiptensorCreatePlan` 的 pref 参数可传 `NULL`，此时使用默认配置。

#### 8.7.1 为什么需要 PlanPreference

```
同一个张量操作，底层可能有几十个候选内核：

┌──────────────────────────────────────────────┐
│ Kernel 1: 大块策略，适合大矩阵                 │
│ Kernel 2: 小块策略，适合小矩阵                 │
│ Kernel 3: 优化了 FP16 数据类型                 │
│ ...                                          │
└──────────────────────────────────────────────┘

→ PlanPreference 让用户控制"怎么选"
```

#### 8.7.2 结构体定义

```cpp
struct hiptensorPlanPreference
{
    hiptensorAutotuneMode_t mAutotuneMode;      // 自动调优模式
    hiptensorCacheMode_t    mCacheMode;         // 缓存模式
    hiptensorAlgo_t          mAlgo;             // 算法选择
    hiptensorJitMode_t       mJitMode;          // JIT 模式
    hiptensorKernelRank_t    mKernelRank;       // 内核秩
    uint32_t               mIncrementalCount;   // 增量计数
};
```

#### 8.7.3 关键字段说明

| 字段 | 说明 | 常用值 |
|------|------|--------|
| **mAlgo** | 算法选择策略 | `DEFAULT` / `DEFAULT_PATIENT` |
| **mCacheMode** | 是否使用 PlanCache | `PEDANTIC`(推荐) / `NONE` |
| **mAutotuneMode** | 候选剪枝策略 | `PRUNE` / `NO_PRUNE` |

**mAlgo 详解**：

| 值 | 说明 | 适用场景 |
|---|------|---------|
| `DEFAULT` | 快速启发式选择 | 大多数情况 |
| `DEFAULT_PATIENT` | 更精确但耗时 | 追求最优性能 |

**mCacheMode 详解**：

| 值 | 说明 |
|---|------|
| `NONE` | 每次重新选择内核 |
| `PEDANTIC` | 使用缓存（推荐） |

**mAutotuneMode 详解**：

| 值 | 说明 |
|---|------|
| `PRUNE` | 快速排除不合适的候选 |
| `NO_PRUNE` | 评估所有候选 |

#### 8.7.4 使用示例

```cpp
// 简单用法：传 NULL 使用默认配置
hiptensorCreatePlan(handle, &plan, opDesc, NULL, workspaceSize);

// 高级用法：自定义偏好
hiptensorPlanPreference_t pref;
hiptensorCreatePlanPreference(handle, &pref,
    HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE);

// 设置使用缓存
hiptensorCacheMode_t cacheMode = HIPTENSOR_CACHE_MODE_PEDANTIC;
hiptensorPlanPreferenceSetAttribute(handle, pref,
    HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cacheMode, sizeof(cacheMode));

hiptensorCreatePlan(handle, &plan, opDesc, pref, workspaceSize);
```

#### 8.7.5 推荐配置

| 场景 | mAlgo | mCacheMode | 说明 |
|------|-------|------------|------|
| **快速启动** | `DEFAULT` | `PEDANTIC` | 大多数情况 |
| **最优性能** | `DEFAULT_PATIENT` | `PEDANTIC` | 首次运行耗时，后续复用缓存 |
| **调试** | `DEFAULT` | `NONE` | 每次重新选择 |

**本质**：在"选择速度"和"内核质量"之间权衡。

### 8.7.6 mSelectionAlgorithm 字段深度分析

`mSelectionAlgorithm`（对应 API 中的 `algo` 参数）是控制**内核选择策略**的核心字段，决定了如何从几十个候选内核中选出最优的一个。

#### 枚举定义

```cpp
// library/include/hiptensor/hiptensor_types.h:164-174
typedef enum
{
    HIPTENSOR_ALGO_ACTOR_CRITIC = -8,  // 使用 Actor-Critic 机器学习模型
    HIPTENSOR_ALGO_DEFAULT = -1,       // 快速启发式选择
    HIPTENSOR_ALGO_DEFAULT_PATIENT = -6, // 更精确但耗时的模型
} hiptensorAlgo_t;
```

#### 三种算法的核心区别

| 算法 | 选择方法 | 选择速度 | 精度 | 适用场景 |
|------|---------|---------|------|---------|
| **DEFAULT** | `bruteForceModel` 遍历测试 | 中等 | 高 | 大多数情况（推荐） |
| **DEFAULT_PATIENT** | `bruteForceModel` 更多测试次数 | 慢 | 最高 | 追求极致性能 |
| **ACTOR_CRITIC** | 机器学习模型预测 | 快 | 中等 | 快速启动场景 |

#### bruteForceModel 详解（DEFAULT / DEFAULT_PATIENT）

**工作原理**：实际运行每个候选内核进行性能测试

```cpp
// library/src/contraction/contraction_selection.cpp
hiptensorStatus_t bruteForceModel(ContractionSolution** winner,
                                  std::vector<ContractionSolution*>& candidates, ...)
{
    // 1. 分配测试用的 GPU 内存
    CHECK_HIP_ALLOC(hipMalloc(&A_d, sizeA));
    CHECK_HIP_ALLOC(hipMalloc(&B_d, sizeB));
    CHECK_HIP_ALLOC(hipMalloc(&D_d, sizeD));
    CHECK_HIP_ALLOC(hipMalloc(&E_d, sizeE));
    CHECK_HIP_ALLOC(hipMalloc(&wspace, workspaceSize));

    // 2. 遍历每个候选内核
    for(auto* solution : candidates)
    {
        // 实际执行内核并计时
        auto [errorCode, time] = (*solution)(
            &alpha, A_d, B_d, &beta, D_d, E_d,
            lengths, strides, modes, ...
            StreamConfig{
                nullptr,           // stream
                true,              // time_kernel = true（计时）
                0,                 // log_level
                options->coldRuns(),  // 预热次数
                options->hotRuns(),   // 正式计时次数
            });

        if(errorCode == SUCCESS && time > 0)
        {
            // 计算性能指标
            auto flops = 2 * m * n * k;  // 每个元素 K 次乘法 + K 次加法
            auto bytes = solution->problemBytes();

            PerfMetrics metrics = {
                solution->uid(),       // 内核ID
                solution->kernelName(), // 内核名称
                time,                   // 平均执行时间
                flops / (1e9 * time),   // TFLOPS
                bytes / (1e6 * time)    // 带宽
            };

            if(metrics > bestMetrics)
                bestSolution = solution;  // 更新最优
        }
    }

    // 3. 按性能排序候选（用于后续 autotune）
    std::sort(candidates by time);

    // 4. 返回最优内核
    *winner = bestSolution;
}
```

#### bruteForceModel 的"假数据"测试

bruteForceModel 在测试时使用的不是用户的真实数据，而是专门准备的测试数据：

```cpp
// 1. 张量数据：未初始化的 GPU 内存
void *A_d, *B_d, *D_d, *E_d;

hipMalloc(&A_d, sizeA);  // 只分配，未初始化（内容是垃圾数据）
hipMalloc(&B_d, sizeB);
hipMalloc(&D_d, sizeD);
hipMalloc(&E_d, sizeE);

// 2. 标量参数：特定值
ScalarData alpha = 1.02;  // 避免 0 或 1
ScalarData beta  = 1.03;
```

**为什么用 1.02, 1.03 而不是 0 或 1？**

| 值 | 问题 |
|---|---|
| alpha = 0 | C = 0*(A*B) + beta*C，跳过矩阵乘法 |
| alpha = 1 | 可能触发编译器优化（1*x = x） |
| beta = 0 | 跳过加法操作 |

使用 1.02/1.03 这种"普通值"确保所有计算步骤都会执行。

**为什么未初始化内存也能测试性能？**

性能测试关注的是**执行时间**，而非计算结果正确性：

| 影响性能的因素（与数据值无关） | 不影响性能的因素 |
|---|---|
| 内存访问模式（连续/非连续） | 输入数据的具体值 |
| 计算量（FLOPS） | 输出结果是否正确 |
| 内存带宽利用率 | - |
| GPU 资源占用（寄存器、共享内存） | - |

**潜在局限性**：

| 局限 | 影响 |
|------|------|
| 内存未初始化 | 可能触发 GPU ECC 错误（如果有） |
| 不测试正确性 | 选中"最快"的内核可能有 bug |
| 缓存状态不同 | 测试内存是新分配的，真实数据可能已缓存 |
| 数据模式不同 | 特殊数据（全 0、稀疏）可能触发不同优化路径 |

**与 INCREMENTAL 的对比**：

| 对比项 | bruteForceModel | INCREMENTAL |
|------|-----------------|-------------|
| 数据来源 | 假数据（未初始化内存） | 用户真实数据 |
| 执行时机 | 创建 Plan 时 | 真实执行时 |
| 测试次数 | 集中一次性 | 分散多次 |
| 准确性 | 可能有偏差 | 真实场景准确 |

**关键特点**：
- **实际执行**：不是理论估算，而是真跑每个内核
- **多次测试**：coldRuns（预热）+ hotRuns（计时），取平均值
- **性能指标**：执行时间、TFLOPS、带宽综合评估

#### actorCriticModel 详解（ACTOR_CRITIC）

**工作原理**：使用预训练的机器学习模型，根据问题特征直接预测最优内核

```cpp
// library/src/contraction/contraction_selection.cpp
template <...>
struct ActorCriticSelection
{
    static hiptensorStatus_t selectWinner(ContractionSolution** winner,
                                          std::unordered_map<size_t, ContractionSolution*> const& candidates,
                                          ...)
    {
        // 根据张量 rank 选择预训练的内核 UID
        auto rank = getRank(a_ms_ks_strides);
        size_t unique_id = 0;

        if(options->isColMajorStrides())
        {
            // 列主序布局的预训练 UID
            if(rank == 1)      unique_id = 16046312426561516674ull;
            else if(rank == 2) unique_id = 5651259715737336589ull;
            else if(rank == 3) unique_id = 5651259715737336589ull;
            // ...
        }
        else
        {
            // 行主序布局的预训练 UID
            if(rank == 1)      unique_id = 9021620837589482599ull;
            else if(rank == 2) unique_id = 6053663486226699267ull;
            // ...
        }

        // 直接查找预训练的最优内核
        if(auto candidate = candidates.find(unique_id); candidate != candidates.end())
        {
            *winner = candidate->second;  // 直接返回，无需测试
            return SUCCESS;
        }
        return EXECUTION_FAILED;
    }
};
```

**关键特点**：
- **无需测试**：直接根据问题特征查找预训练的内核 ID
- **速度极快**：O(1) 查找，没有运行开销
- **依赖训练**：模型需要预先训练，可能不适应所有场景

#### 选择流程对比

```
┌─────────────────────────────────────────────────────────────────┐
│                     内核选择流程对比                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DEFAULT / DEFAULT_PATIENT:                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. 获取所有候选内核（20-50 个）                           │   │
│  │ 2. 遍历每个候选：                                         │   │
│  │    - 分配测试内存                                         │   │
│  │    - coldRuns 次预热运行                                   │   │
│  │    - hotRuns 次正式计时                                   │   │
│  │    - 计算性能指标                                         │   │
│  │ 3. 按性能排序                                             │   │
│  │ 4. 返回最优内核                                           │   │
│  │ 5. 释放测试内存                                           │   │
│  │ 耗时：50 内核 × 6 次运行 × 2ms ≈ 600ms                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ACTOR_CRITIC:                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. 计算问题特征（rank、布局等）                           │   │
│  │ 2. 查表获取预训练的内核 UID                               │   │
│  │ 3. 从候选列表中查找对应内核                               │   │
│  │ 4. 直接返回                                               │   │
│  │ 耗时：< 1ms                                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 与 PlanCache 的协同工作

无论使用哪种算法，选择结果都可以被缓存：

```cpp
// 缓存命中时，跳过所有算法选择
if(planCache && cacheMode == PEDANTIC)
{
    auto Uid = planCache->querySolutionUid(desc);
    if(Uid > 0)
    {
        winner = findSolutionByUid(candidates, Uid);  // 直接使用缓存
    }
}

// 缓存未命中时，根据 mSelectionAlgorithm 选择
if(result != SUCCESS)
{
    if(algo == DEFAULT || algo == DEFAULT_PATIENT)
        bruteForceModel(&winner, candidates, ...);
    else if(algo == ACTOR_CRITIC)
        actorCriticModel(&winner, candidates, ...);

    // 缓存选择结果
    planCache->addCacheLine(hashId, winner->uid());
}
```

**性能对比**：

| 场景 | DEFAULT | ACTOR_CRITIC | 有缓存 |
|------|---------|--------------|--------|
| 首次运行 | ~600ms | <1ms | - |
| 再次运行 | ~600ms | <1ms | <1ms |
| 缓存命中 | <1ms | <1ms | <1ms |

#### 使用建议

| 场景 | 推荐 algo | mCacheMode | 说明 |
|------|----------|------------|------|
| **生产环境（推荐）** | `DEFAULT` | `PEDANTIC` | 首次运行耗时，后续快速 |
| **追求极致性能** | `DEFAULT_PATIENT` | `PEDANTIC` | 更多测试次数，更准确 |
| **快速启动** | `ACTOR_CRITIC` | `PEDANTIC` | 启动快，但可能不是最优 |
| **调试/测试** | `DEFAULT` | `NONE` | 每次重新选择 |

**本质**：`mSelectionAlgorithm` 决定了**时间换精度**还是**速度换精度**。

### 8.7.7 mKernelRank 字段深度分析

`mKernelRank` 是 `hiptensorPlanPreference` 中一个特殊的字段，目前处于**预留但未实际使用**的状态。

#### 结构体中的位置

```cpp
// library/src/include/data_types.hpp:115-127
struct hiptensorPlanPreference
{
    hiptensorAutotuneMode_t mAutotuneMode;
    hiptensorCacheMode_t    mCacheMode;
    int32_t                 mIncrementalCount;
    int32_t                 mKernelRank;      // <-- KernelRank 字段
    hiptensorJitMode_t      mJit;

    hiptensorAlgo_t mSelectionAlgorithm;
    std::vector<void*> mCandidates;
    void*              mSolution;
};
```

#### 含义：张量维度数

根据文档 `docs/conceptual/programmers-guide.rst:70-82`，**Tensor rank 指的是张量的维度数**。在张量缩并操作中，维度被分为三类：

| 维度类型 | 含义 | 出现位置 |
|---------|------|---------|
| **M 维度** | 只属于输入 A 的维度 | A[M0...Mn, K0...Kn] 和 C/D[M0...Mn, N0...Nn] |
| **N 维度** | 只属于输入 B 的维度 | B[N0...Nn, K0...Kn] 和 C/D[M0...Mn, N0...Nn] |
| **K 维度** | 收缩维度（求和维度） | 同时出现在 A 和 B 中 |

**维度示例**：

```
例如矩阵乘法 C = A × B:
- A: M×K
- B: K×N
- C: M×N
- 每个 M/N/K 类别有 1 个维度

例如高维张量缩并:
- A[M0..M5, K0..K5] - 6个M维度 + 6个K维度
- B[N0..N5, K0..K5] - 6个N维度 + 6个K维度
- C[M0..M5, N0..N5] - 6个M维度 + 6个N维度
- 这是 M6N6K6 配置，总共 rank 12
```

#### 代码中的使用

**1. 设置接口**

```cpp
// library/src/hiptensor.cpp:590-592
case HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK:
    std::memcpy(&pref->mKernelRank, buf, sizeInBytes);
    break;
```

**2. 默认值**

```cpp
// library/src/hiptensor.cpp:554
(*pref)->mKernelRank = 0;
```

**3. 当前状态**

`mKernelRank` 目前只是被存储，**并未在 kernel 选择逻辑中实际使用**。

证据：
- 只在设置时被赋值
- 搜索整个代码库，没有任何地方读取或使用 `pref->mKernelRank`
- 所有测试和示例代码都没有使用 `HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK`

#### 与 Kernel 实例的关系

实际的 kernel 实例通过模板参数 `NumDimsM`, `NumDimsN`, `NumDimsK` 在编译时确定：

```cpp
// library/src/contraction/contraction_meta_traits.hpp:40-42
#define MaxNumDimsM 6
#define MaxNumDimsN 6
#define MaxNumDimsK 6
```

每个具体的 kernel 实例（如 `device_contraction_bilinear_m6_n6_k6_xdl_c_shuffle_*`）都有固定的 M/N/K 维度配置。

Kernel 选择时使用的维度信息来自 `ContractionSolutionParams`：

```cpp
// library/src/contraction/contraction_solution_params.hpp:46-48
virtual int32_t dimsM() const = 0;
virtual int32_t dimsN() const = 0;
virtual int32_t dimsK() const = 0;
```

#### 设计问题：为什么只有一个字段？

**理论上应该有三个字段来分别控制 M/N/K 维度偏好**，但当前只有一个 `mKernelRank`。

**可能的原因**：

1. **预留但未完成的功能** - 这是一个预留的 API 接口，实现尚未完成
2. **预留接口兼容性** - 可能是为了与 cuTENSOR 兼容而预留

**可能的完整设计（未实现）**：

```cpp
// 理论上应该有的设计：
struct hiptensorPlanPreference
{
    // ... 其他字段 ...
    int32_t mKernelRankM;  // M 维度数偏好
    int32_t mKernelRankN;  // N 维度数偏好
    int32_t mKernelRankK;  // K 维度数偏好
    // ... 其他字段 ...
};
```

#### 总结

```
当前状态：
┌─────────────────────────────────────────┐
│ mKernelRank = 0 (默认值，从未被使用)     │
│ Kernel 选择依据：ContractionSolutionParams │
│                  的 dimsM/N/K() 方法      │
└─────────────────────────────────────────┘

可能的完整设计（未实现）：
┌─────────────────────────────────────────┐
│ mKernelRankM - 指定偏好 M 维度数          │
│ mKernelRankN - 指定偏好 N 维度数          │
│ mKernelRankK - 指定偏好 K 维度数          │
│ → 用于过滤/优先选择特定 rank 的 kernel    │
└─────────────────────────────────────────┘
```

**结论**：`mKernelRank` 是 `hiptensorPlanPreference` 的一个预留字段，用于指定张量操作的维度数偏好。它通过 `HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK` 枚举值设置，但目前在代码中主要是预留接口，尚未在实际的 kernel 选择算法中使用。正确的实现应该有三个字段分别对应 M、N、K 维度。

### 8.7.8 mAutotuneMode 字段深度分析

`mAutotuneMode` 控制是否在**真实运行时**继续优化内核选择。

#### 枚举定义

```cpp
// library/include/hiptensor/hiptensor_types.h:233-237
typedef enum
{
    HIPTENSOR_AUTOTUNE_MODE_NONE        = 0,  // 禁用自动调优
    HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL = 1,  // 增量式自动调优
} hiptensorAutotuneMode_t;
```

#### 两种模式对比

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| **NONE** | 信任创建 Plan 时的选择结果 | 通用场景（推荐） |
| **INCREMENTAL** | 前 N 次真实运行中探索不同内核 | 追求极致性能 |

#### INCREMENTAL 模式工作原理

```
假设 mIncrementalCount = 5, mCandidates.size() = 10

第 1 次执行: callCount=0 → 使用 Candidates[0], 记录时间
第 2 次执行: callCount=1 → 使用 Candidates[1], 记录时间
第 3 次执行: callCount=2 → 使用 Candidates[2], 记录时间
第 4 次执行: callCount=3 → 使用 Candidates[3], 记录时间
第 5 次执行: callCount=4 → 使用 Candidates[4], 记录时间
            callCount >= mIncrementalCount
            → 将 bestSolution 存入 PlanCache

后续执行: PlanCache 命中 → 直接使用缓存内核
```

#### 关键代码

```cpp
// library/src/include/plancache_autotune.hpp

// setAutotune() - 选择本轮要执行的内核
void setAutotune(...)
{
    auto Uid = planCache->querySolutionUid(plan->mOpDesc);

    if (Uid > 0)
    {
        // PlanCache 命中，直接使用缓存
        plan->mSolution = findSolutionByUid(candidates, Uid);
    }
    else if (mAutotuneMode == INCREMENTAL)
    {
        // PlanCache 未命中，按顺序探索候选内核
        if (callCount < mIncrementalCount && callCount < mCandidates.size())
            plan->mSolution = mCandidates[callCount];
    }
}

// saveAutotune() - 保存本轮执行结果
void saveAutotune(float time, ...)
{
    if (mAutotuneMode == INCREMENTAL)
    {
        // 记录最优结果
        if (time < minTime)
            mBestSolution = {time, currentSolution};

        mCallCount++;

        // 达到探索次数，存入缓存
        if (mCallCount >= mIncrementalCount)
        {
            planCache->addCacheLine(hashID, mBestSolution->uid());
            mCallCount = 0;  // 重置状态
        }
    }
}
```

### 8.7.9 bruteForceModel 测试机制详解

`bruteForceModel` 在**创建 Plan 时**执行，用于测试并选择最优内核。

#### "假数据"的真相

```cpp
// library/src/contraction/contraction_selection.cpp

// 1. 分配测试内存（未初始化！）
void *A_d, *B_d, *D_d, *E_d;
hipMalloc(&A_d, sizeA);  // 垃圾数据
hipMalloc(&B_d, sizeB);  // 垃圾数据
hipMalloc(&D_d, sizeD);  // 垃圾数据
hipMalloc(&E_d, sizeE);  // 垃圾数据

// 2. 只有标量参数有具体值
ScalarData alpha = ScalarData(computeType, 1.02);
ScalarData beta  = ScalarData(computeType, 1.03);

// 3. 遍历所有候选内核并计时
for (auto* solution : candidates)
{
    auto [errorCode, time] = (*solution)(&alpha, A_d, B_d, &beta, D_d, E_d, ...);
    // 计算性能指标 TFLOPS
    if (tflops > best_tflops)
        bestSolution = solution;
}
```

| 数据 | 来源 | 内容 |
|------|------|------|
| **A_d, B_d** | hipMalloc，未初始化 | GPU 内存中的随机数据 |
| **D_d, E_d** | hipMalloc，未初始化 | GPU 内存中的随机数据 |
| **alpha, beta** | 代码设置 | 1.02, 1.03（避免特殊值） |

#### 为什么"假数据"也能测试性能？

**性能测试关注执行时间，不关心计算结果正确性**

```
影响性能的因素（与数据值无关）：
├── 内存访问模式（连续/非连续）
├── 计算量（FLOPS）
├── 内存带宽利用率
└── GPU 资源占用（寄存器、共享内存）

不影响性能的因素：
├── 输入数据的具体值
└── 输出结果是否正确
```

#### 为什么 alpha/beta 用 1.02, 1.03？

```cpp
alpha = 0  → C = 0 * (A*B) + beta*C（跳过矩阵乘法！）
alpha = 1  → 可能触发编译器优化（1*x = x）
beta = 0   → C = alpha * A * B（跳过加法！）

// 1.02, 1.03 这种"普通值"确保所有计算步骤都会执行
```

### 8.7.10 内核选择完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    hiptensorCreatePlan() 流程                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 检查 PlanCache 是否命中                                                  │
│     if (cacheMode == PEDANTIC && planCache->query(desc) 命中)               │
│         → 直接使用缓存内核，跳过以下所有步骤                                  │
│                                                                             │
│  2. PlanCache 未命中，执行内核选择算法                                        │
│     if (algo == DEFAULT || DEFAULT_PATIENT)                                 │
│         → bruteForceModel(): 使用假数据测试所有候选，选出最快的               │
│     else if (algo == ACTOR_CRITIC)                                          │
│         → actorCriticModel(): 使用预训练模型直接预测                          │
│                                                                             │
│  3. 保存结果                                                                 │
│     pref->mSolution = winner              // 选中的内核                      │
│     pref->mCandidates = all_candidates    // 用于 INCREMENTAL 探索          │
│                                                                             │
│  4. 如果 autotuneMode == NONE                                                │
│     → 将 pref->mSolution 存入 PlanCache                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    hiptensorContract() 执行流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  setAutotune():                                                              │
│     if (PlanCache 命中) → 使用缓存内核（跳过 autotune）                       │
│     elif (INCREMENTAL && callCount < mIncrementalCount)                      │
│         → 使用 mCandidates[callCount]                                        │
│                                                                             │
│  [执行内核，使用用户的真实数据，返回执行时间]                                  │
│                                                                             │
│  saveAutotune(time):                                                         │
│     if (INCREMENTAL)                                                         │
│         → 记录性能，更新 bestSolution                                         │
│         → callCount++                                                        │
│         → if (callCount >= mIncrementalCount)                                │
│             → 将 bestSolution 存入 PlanCache                                  │
│             → 重置状态，下一轮重新开始                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.7.11 PlanPreference 推荐配置汇总

| 场景 | algo | autotuneMode | mIncrementalCount | cacheMode | 说明 |
|------|------|--------------|-------------------|-----------|------|
| **生产环境（推荐）** | DEFAULT | NONE | - | PEDANTIC | 首次创建 Plan 耗时，后续快速 |
| **追求极致性能** | DEFAULT | INCREMENTAL | 5-10 | PEDANTIC | 前 N 次真实运行优化，后续复用 |
| **快速启动** | ACTOR_CRITIC | NONE | - | PEDANTIC | 启动快，但可能不是最优 |
| **调试/测试** | DEFAULT | NONE | - | NONE | 每次重新选择 |

```
关键协作关系：
┌─────────────────────────────────────────────────────────────────────────────┐
│  PlanCache 命中 → 直接使用缓存，跳过 algo 和 autotune                         │
│  PlanCache 未命中 → algo 选择 → autotune 优化 → 存入 PlanCache               │
│                                                                             │
│  bruteForceModel (创建 Plan 时):                                             │
│    - 使用假数据（未初始化内存 + 1.02/1.03 标量）                              │
│    - 目标：快速筛选候选内核                                                   │
│                                                                             │
│  INCREMENTAL (真实运行时):                                                    │
│    - 使用用户的真实数据                                                       │
│    - 目标：在真实场景中精细选择最优内核                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.8 hiptensorPlan_t 结构详解

**Plan** 是执行计划，包含操作描述符和选中的最优内核。

#### 8.8.1 结构体定义

```cpp
struct hiptensorPlan
{
    uint64_t                       mRequiredWorkspace;  // 工作空间大小
    hiptensorOperationDescriptor_t mOpDesc;             // 操作描述符
    hiptensorPlanPreference_t      mPref;               // 偏好设置（含选中内核）
};
```

`mPref` 中的关键字段：
```cpp
struct hiptensorPlanPreference
{
    ...
    std::vector<void*> mCandidates;  // 候选内核列表
    void*              mSolution;    // 选中的最优内核！
};
```

#### 8.8.2 Plan 创建流程

```
┌─────────────────────────────────────────────────────────────┐
│                   contractionInitPlan()                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 获取所有候选内核（可能有几十个）                          │
│     pref->mCandidates = allSolutions();                     │
│                                                             │
│  2. 根据条件过滤：                                           │
│     - 操作类型 (BILINEAR/SCALE/...)                         │
│     - 数据类型 (A/B/C/D 的类型)                              │
│     - 操作符 (IDENTITY/UNARY)                               │
│                                                             │
│  3. 检查 PlanCache 缓存                                     │
│     if (cacheMode == PEDANTIC) {                            │
│         Uid = planCache->querySolutionUid(desc);            │
│         if (命中) winner = findByUid(Uid);                  │
│     }                                                       │
│                                                             │
│  4. 缓存未命中 → 运行选择模型                                 │
│     - bruteForceModel：遍历评估所有候选                      │
│     - actorCriticModel：机器学习模型选择                     │
│                                                             │
│  5. 保存选中的内核                                           │
│     pref->mSolution = winner;  ← 核心结果！                  │
│                                                             │
│  6. 缓存选择结果                                             │
│     planCache->addCacheLine(hashId, winner->uid());         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 8.8.3 Plan 包含的内容

| 内容 | 存储位置 | 说明 |
|------|---------|------|
| 操作描述符 | `plan->mOpDesc` | 描述要执行的操作 |
| 候选内核 | `plan->mPref->mCandidates` | 过滤后的候选列表 |
| **最优内核** | `plan->mPref->mSolution` | **选中的内核（核心！）** |
| 工作空间 | `plan->mRequiredWorkspace` | 所需内存大小 |

#### 8.8.4 内核选择策略

| 策略 | 函数 | 说明 |
|------|------|------|
| 缓存优先 | `querySolutionUid()` | 如果缓存命中，直接使用 |
| BruteForce | `bruteForceModel()` | 遍历所有候选，评估性能 |
| Actor-Critic | `actorCriticModel()` | 机器学习模型选择 |

#### 8.8.5 与执行的关系

```
hiptensorContract(plan, ...)
        │
        ▼
winner = plan->mPref->mSolution;  // 获取选中的内核
winner->run(...);                  // 执行
```

**本质**：Plan 创建的核心是**内核选择**——从几十个候选中选出最优的 `mSolution`，后续执行时直接调用它。

### 8.9 内核选择机制详解

#### 8.9.1 什么是内核

**内核**是实际在 GPU 上执行的函数。对于同一个数学运算（如矩阵乘法），可以有多种不同的实现方式：

```
┌─────────────────────────────────────────────────────────────┐
│              同一个操作：C = A × B (矩阵乘法)                  │
├─────────────────────────────────────────────────────────────┤
│  Kernel 1: 大块策略 (256×256 块) - 适合大矩阵                 │
│  Kernel 2: 小块策略 (32×32 块) - 适合小矩阵                   │
│  Kernel 3: 批量优化 - 适合同时计算多个小矩阵                   │
│  Kernel 4: FP16 优化 - 利用 GPU 的 FP16 专用硬件              │
│  ... 还有几十个其他变体                                       │
└─────────────────────────────────────────────────────────────┘
```

#### 8.9.2 为什么需要选择

不同内核在不同场景下性能差异巨大：

| 场景 | Kernel A (大块) | Kernel B (小块) | 差异 |
|------|----------------|-----------------|------|
| 4096×4096 | 2.5ms | 15.0ms | 6倍 |
| 64×64 | 0.8ms | 0.1ms | 8倍 |

**没有"万能内核"**——必须根据实际情况选择。

#### 8.9.3 bruteForceModel 详解

bruteForceModel 是内核选择的核心函数，**实际执行每个候选内核进行性能评估**：

```cpp
// 源码：contraction_selection.cpp
hiptensorStatus_t bruteForceModel(ContractionSolution** winner,
                                  std::vector<ContractionSolution*>& candidates,
                                  ...)
{
    // 1. 分配测试用的 GPU 内存
    hipMalloc(&A_d, sizeA);
    hipMalloc(&B_d, sizeB);
    hipMalloc(&D_d, sizeD);
    hipMalloc(&E_d, sizeE);

    // 2. 遍历每个候选内核，实际执行并计时
    for(auto* solution : candidates)
    {
        // 实际执行内核，返回执行时间
        auto [errorCode, time] = (*solution)(
            &alpha, A_d, B_d, &beta, D_d, E_d,
            lengths, strides, modes, ...
            StreamConfig{
                nullptr,       // stream
                true,          // time_kernel = true（计时）
                options->coldRuns(),  // 预热次数
                options->hotRuns(),   // 正式计时次数
            });

        // 计算性能指标：TFLOPS、带宽
        if(errorCode == SUCCESS && time > 0) {
            metrics = {uid, name, time, tflops, bandwidth};
            if(metrics > bestMetrics) {
                bestSolution = solution;  // 更新最优
            }
        }
    }

    // 3. 按性能排序候选（用于后续 autotune）
    sort(candidates by time);

    // 4. 返回最优内核
    *winner = bestSolution;
}
```

**关键点**：
- **实际执行**每个候选内核（非模型估算）
- **多次运行**取平均值（coldRuns + hotRuns）
- **计算性能指标**：执行时间、TFLOPS、带宽

#### 8.9.4 执行次数配置

通过环境变量控制测试次数：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `HIPTENSOR_COLD_RUNS` | 1 | 预热次数（不计入统计） |
| `HIPTENSOR_HOT_RUNS` | 5 | 正式计时次数 |

```bash
# 设置更多测试次数以获得更准确的结果
export HIPTENSOR_COLD_RUNS=3
export HIPTENSOR_HOT_RUNS=10
```

#### 8.9.5 首次运行的开销

```
首次运行 hiptensorCreatePlan():
┌─────────────────────────────────────────────────────────────┐
│ 候选内核数量: 50个                                           │
│ 每个内核执行: 1(cold) + 5(hot) = 6次                         │
│ 假设每次: 2ms                                                │
│ 总耗时: 50 × 6 × 2ms = 600ms                                │
│                                                             │
│ ↓ 缓存选择结果                                               │
│                                                             │
│ 再次运行相同操作:                                             │
│ 查缓存: ~1ms                                                 │
│ 总耗时: 1ms                                                  │
│ 加速: 600倍!                                                 │
└─────────────────────────────────────────────────────────────┘
```

#### 8.9.6 与执行的关系

```
hiptensorCreatePlan() 时：
    winner = bruteForceModel(candidates)
    plan->mPref->mSolution = winner
    planCache->add(hash(desc), winner->uid())

hiptensorContract() 时：
    winner = plan->mPref->mSolution
    winner->run(alpha, A, B, beta, C, D, workspace, stream)
```

**本质**：内核选择是 hipTensor 性能优化的核心——实际执行测试，选出最优内核，后续直接调用。

---

## 9. 操作符枚举 (hiptensorOperator_t)

### 9.1 一元操作符

| 枚举值 | 说明 | 数学表达式 |
|--------|------|-----------|
| `HIPTENSOR_OP_IDENTITY` | 恒等操作 | y = x |
| `HIPTENSOR_OP_SQRT` | 平方根 | y = √x |
| `HIPTENSOR_OP_RELU` | ReLU | y = max(0, x) |
| `HIPTENSOR_OP_CONJ` | 共轭 | y = conj(x) |
| `HIPTENSOR_OP_RCP` | 倒数 | y = 1/x |
| `HIPTENSOR_OP_SIGMOID` | Sigmoid | y = 1/(1+e^(-x)) |
| `HIPTENSOR_OP_TANH` | 双曲正切 | y = tanh(x) |
| `HIPTENSOR_OP_EXP` | 指数 | y = e^x |
| `HIPTENSOR_OP_LOG` | 自然对数 | y = ln(x) |
| `HIPTENSOR_OP_ABS` | 绝对值 | y = \|x\| |
| `HIPTENSOR_OP_NEG` | 取负 | y = -x |
| `HIPTENSOR_OP_SIN/COS/TAN` | 三角函数 | y = sin(x)/cos(x)/tan(x) |
| `HIPTENSOR_OP_SINH/COSH` | 双曲函数 | y = sinh(x)/cosh(x) |
| `HIPTENSOR_OP_ASIN/ACOS/ATAN` | 反三角函数 | - |
| `HIPTENSOR_OP_ASINH/ACOSH/ATANH` | 反双曲函数 | - |
| `HIPTENSOR_OP_CEIL/FLOOR` | 取整函数 | - |

### 9.2 二元操作符

| 枚举值 | 说明 | 数学表达式 |
|--------|------|-----------|
| `HIPTENSOR_OP_ADD` | 加法 | y = a + b |
| `HIPTENSOR_OP_MUL` | 乘法 | y = a * b |
| `HIPTENSOR_OP_MAX` | 最大值 | y = max(a, b) |
| `HIPTENSOR_OP_MIN` | 最小值 | y = min(a, b) |

---

## 10. 算法选择 (hiptensorAlgo_t)

| 枚举值 | 说明 |
|--------|------|
| `HIPTENSOR_ALGO_DEFAULT` | 让内部启发式算法自动选择 |
| `HIPTENSOR_ALGO_DEFAULT_PATIENT` | 使用更精确但耗时的模型 |
| `HIPTENSOR_ALGO_ACTOR_CRITIC` | 使用 Actor-Critic 模型选择 |

---

## 11. 返回值状态码 (hiptensorStatus_t)

| 枚举值 | 值 | 说明 |
|--------|---|------|
| `HIPTENSOR_STATUS_SUCCESS` | 0 | 操作成功 |
| `HIPTENSOR_STATUS_NOT_INITIALIZED` | 1 | 句柄未初始化 |
| `HIPTENSOR_STATUS_ALLOC_FAILED` | 3 | 资源分配失败 |
| `HIPTENSOR_STATUS_INVALID_VALUE` | 7 | 无效参数值 |
| `HIPTENSOR_STATUS_ARCH_MISMATCH` | 8 | 架构不支持或设备未就绪 |
| `HIPTENSOR_STATUS_EXECUTION_FAILED` | 13 | GPU 程序或内核执行失败 |
| `HIPTENSOR_STATUS_INTERNAL_ERROR` | 14 | 内部错误 |
| `HIPTENSOR_STATUS_NOT_SUPPORTED` | 15 | 请求的操作不支持 |
| `HIPTENSOR_STATUS_CK_ERROR` | 17 | Composable Kernels 调用失败 |
| `HIPTENSOR_STATUS_HIP_ERROR` | 18 | 未知 HIP 错误 |
| `HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE` | 19 | 工作空间不足 |
| `HIPTENSOR_STATUS_INSUFFICIENT_DRIVER` | 20 | 驱动版本不足 |
| `HIPTENSOR_STATUS_IO_ERROR` | 21 | 文件 I/O 错误 |

---

## 12. 典型使用流程

### 12.1 基本流程

```cpp
#include <hiptensor/hiptensor.h>

// 1. 创建句柄
hiptensorHandle_t handle;
hiptensorCreate(&handle);

// 2. 创建张量描述符
hiptensorTensorDescriptor_t descA, descB, descC, descD;
hiptensorCreateTensorDescriptor(handle, &descA, ...);
hiptensorCreateTensorDescriptor(handle, &descB, ...);
// ...

// 3. 创建操作描述符（例如收缩）
hiptensorOperationDescriptor_t opDesc;
hiptensorCreateContraction(handle, &opDesc,
    descA, modeA, opA,
    descB, modeB, opB,
    descC, modeC, opC,
    descD, modeD,
    HIPTENSOR_COMPUTE_DESC_32F);

// 4. 估算工作空间
uint64_t workspaceSize;
hiptensorEstimateWorkspaceSize(handle, opDesc, nullptr,
    HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize);

// 5. 创建 Plan
hiptensorPlan_t plan;
hiptensorCreatePlan(handle, &plan, opDesc, nullptr, workspaceSize);

// 6. 分配工作空间
void* d_workspace;
hipMalloc(&d_workspace, workspaceSize);

// 7. 执行操作
float alpha = 1.0f, beta = 0.0f;
hiptensorContract(handle, plan, &alpha, d_A, d_B, &beta, d_C, d_D,
    d_workspace, workspaceSize, stream);

// 8. 清理
hipFree(d_workspace);
hiptensorDestroyPlan(plan);
hiptensorDestroyOperationDescriptor(opDesc);
hiptensorDestroyTensorDescriptor(descA);
// ... 销毁其他描述符
hiptensorDestroy(handle);
```

### 12.2 完整示例：矩阵乘法 (张量收缩)

```cpp
#include <hiptensor/hiptensor.h>
#include <hip/hip_runtime.h>
#include <stdio.h>

#define M 1024
#define N 1024
#define K 1024

int main() {
    // 1. 创建 hipTensor 句柄
    hiptensorHandle_t handle;
    hiptensorStatus_t status = hiptensorCreate(&handle);
    if (status != HIPTENSOR_STATUS_SUCCESS) {
        printf("Failed to create handle: %s\n", hiptensorGetErrorString(status));
        return -1;
    }

    // 2. 定义张量维度和模式
    // 矩阵乘法: C(m,n) = A(m,k) * B(k,n)
    int64_t extentA[] = {M, K};  // A: m x k
    int64_t extentB[] = {K, N};  // B: k x n
    int64_t extentC[] = {M, N};  // C: m x n

    int32_t modeA[] = {0, 2};    // m, k
    int32_t modeB[] = {2, 1};    // k, n
    int32_t modeC[] = {0, 1};    // m, n
    int32_t modeD[] = {0, 1};    // m, n (输出)

    // 3. 创建张量描述符
    hiptensorTensorDescriptor_t descA, descB, descC, descD;

    hiptensorCreateTensorDescriptor(handle, &descA, 2, extentA, NULL,
        HIPTENSOR_R_32F, 4);
    hiptensorCreateTensorDescriptor(handle, &descB, 2, extentB, NULL,
        HIPTENSOR_R_32F, 4);
    hiptensorCreateTensorDescriptor(handle, &descC, 2, extentC, NULL,
        HIPTENSOR_R_32F, 4);
    hiptensorCreateTensorDescriptor(handle, &descD, 2, extentC, NULL,
        HIPTENSOR_R_32F, 4);

    // 4. 创建收缩操作描述符
    hiptensorOperationDescriptor_t opDesc;
    status = hiptensorCreateContraction(
        handle, &opDesc,
        descA, modeA, HIPTENSOR_OP_IDENTITY,
        descB, modeB, HIPTENSOR_OP_IDENTITY,
        descC, modeC, HIPTENSOR_OP_IDENTITY,
        descD, modeD,
        HIPTENSOR_COMPUTE_DESC_32F);

    // 5. 创建 Plan Preference
    hiptensorPlanPreference_t planPref;
    hiptensorCreatePlanPreference(handle, &planPref,
        HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_DEFAULT);

    // 6. 估算工作空间
    uint64_t workspaceSize;
    hiptensorEstimateWorkspaceSize(handle, opDesc, planPref,
        HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize);
    printf("Required workspace: %lu bytes\n", workspaceSize);

    // 7. 创建 Plan
    hiptensorPlan_t plan;
    hiptensorCreatePlan(handle, &plan, opDesc, planPref, workspaceSize);

    // 8. 分配设备内存
    float *d_A, *d_B, *d_C, *d_D, *d_workspace;
    hipMalloc(&d_A, M * K * sizeof(float));
    hipMalloc(&d_B, K * N * sizeof(float));
    hipMalloc(&d_C, M * N * sizeof(float));
    hipMalloc(&d_D, M * N * sizeof(float));
    hipMalloc(&d_workspace, workspaceSize);

    // 初始化数据（省略）...

    // 9. 执行张量收缩
    float alpha = 1.0f, beta = 0.0f;
    hipStream_t stream;
    hipStreamCreate(&stream);

    status = hiptensorContract(handle, plan, &alpha, d_A, d_B, &beta,
        d_C, d_D, d_workspace, workspaceSize, stream);

    hipStreamSynchronize(stream);

    // 10. 清理资源
    hipStreamDestroy(stream);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipFree(d_D);
    hipFree(d_workspace);

    hiptensorDestroyPlan(plan);
    hiptensorDestroyPlanPreference(planPref);
    hiptensorDestroyOperationDescriptor(opDesc);
    hiptensorDestroyTensorDescriptor(descA);
    hiptensorDestroyTensorDescriptor(descB);
    hiptensorDestroyTensorDescriptor(descC);
    hiptensorDestroyTensorDescriptor(descD);
    hiptensorDestroy(handle);

    printf("Tensor contraction completed successfully!\n");
    return 0;
}
```

### 12.3 张量归约示例

```cpp
// 归约示例: 对张量的某些维度求和
// 输入: A(m,n,k) -> 输出: D(m,k)  (对 n 维度归约)

int64_t extentA[] = {M, N, K};  // 3D 张量
int64_t extentD[] = {M, K};     // 2D 输出

int32_t modeA[] = {0, 1, 2};    // m, n, k
int32_t modeD[] = {0, 2};       // m, k (n 被归约)

// 创建归约操作
hiptensorCreateReduction(
    handle, &opDesc,
    descA, modeA, HIPTENSOR_OP_IDENTITY,
    descD, modeD, HIPTENSOR_OP_IDENTITY,  // C 与 D 相同
    descD, modeD,
    HIPTENSOR_OP_ADD,                     // 使用加法归约
    HIPTENSOR_COMPUTE_DESC_32F);

// 执行归约
hiptensorReduce(handle, plan, &alpha, d_A, &beta, d_C, d_D,
    d_workspace, workspaceSize, stream);
```

### 12.4 张量置换示例

```cpp
// 置换示例: 转置操作
// 输入: A(m,n) -> 输出: B(n,m)

int64_t extentA[] = {M, N};
int64_t extentB[] = {N, M};

int32_t modeA[] = {0, 1};  // m, n
int32_t modeB[] = {1, 0};  // n, m (交换维度)

// 创建置换操作
hiptensorCreatePermutation(
    handle, &opDesc,
    descA, modeA, HIPTENSOR_OP_IDENTITY,
    descB, modeB,
    HIPTENSOR_COMPUTE_DESC_32F);

// 执行置换
hiptensorPermute(handle, plan, &alpha, d_A, d_B, stream);
```

---

## 13. 与 cuTensor 的对应关系

hipTensor API 设计与 NVIDIA cuTensor 高度兼容：

| 操作类型 | hipTensor | cuTensor | 说明 |
|---------|-----------|----------|------|
| 张量收缩 | `hiptensorContract` | `cutensorContract` | D = α*A*B + β*C |
| 张量归约 | `hiptensorReduce` | `cutensorReduce` | D = α*reduce(A) + β*C |
| 张量置换 | `hiptensorPermute` | `cutensorPermute` | 维度重排 |
| 逐元素二元 | `hiptensorElementwiseBinaryExecute` | `cutensorElementwiseBinaryExecute` | 逐元素操作 |
| 逐元素三元 | `hiptensorElementwiseTrinaryExecute` | `cutensorElementwiseTrinaryExecute` | 三张量逐元素操作 |

---

## 14. API 总览

hipTensor 对外暴露约 **40+ 个公开 API 函数**，涵盖以下类别：

| 类别 | 函数数量 | 主要功能 |
|------|---------|---------|
| 句柄管理 | 7 | 创建/销毁/缓存管理 |
| 张量描述符 | 2 | 张量元数据描述 |
| 操作描述符 | 7 | 收缩/归约/置换/逐元素 |
| Plan 管理 | 6 | 执行计划创建和配置 |
| 执行函数 | 5 | 实际计算执行 |
| 辅助工具 | 4 | 工作空间估算/版本/错误 |
| 日志系统 | 6 | 日志配置和输出 |

---

## 15. FAQ（常见问题）

### Q1: 什么是内核选择？为什么需要它？

**答**：内核（Kernel）是实际在 GPU 上执行的函数。对于同一个数学运算（如矩阵乘法），可以有多种不同的实现方式（大块策略、小块策略、批量优化等）。不同内核在不同场景下性能差异可达数倍甚至数十倍，因此必须根据实际问题的参数（矩阵大小、数据类型、内存布局等）选择最优内核。

### Q2: bruteForceModel 是如何选择最优内核的？

**答**：bruteForceModel **实际执行**每个候选内核进行性能评估，而不是理论估算。流程如下：

```
1. 分配测试用的 GPU 内存（A_d, B_d, D_d, E_d, workspace）
2. 遍历每个候选内核：
   - 预热运行（coldRuns 次，不计入统计）
   - 正式计时（hotRuns 次，取平均值）
   - 计算 TFLOPS 和带宽指标
3. 按性能排序所有候选
4. 返回最优内核
5. 释放测试内存
```

### Q3: coldRuns 和 hotRuns 分别是什么？为什么需要区分？

**答**：

| 类型 | 说明 | 是否计入统计 |
|------|------|-------------|
| **coldRuns（冷启动）** | 首次运行，GPU 缓存未热 | 否 |
| **hotRuns（热启动）** | 缓存已热的后续运行 | 是 |

冷启动时，内核代码需要从主机内存加载到 GPU 指令缓存，GPU 频率可能还未升到最高，页表映射也在建立中，这些开销不应计入性能统计。通过预热运行让 GPU 进入稳定状态后再计时。

默认配置：`coldRuns=0`，`hotRuns=1`。可通过环境变量调整：
```bash
export HIPTENSOR_COLD_RUNS=3   # 预热 3 次
export HIPTENSOR_HOT_RUNS=10   # 计时 10 次取平均
```

### Q4: 性能指标 TFLOPS 和带宽是如何计算的？

**答**：

```cpp
// 问题维度：m × n × k 的矩阵乘法
auto flops = std::size_t(2) * m * n * k;  // 乘法和加法各 m*n*k 次
auto bytes = solution->problemBytes();     // 内存访问字节数

TFLOPS = flops / (10^9 × avg_time_ms × 10^-3)
带宽    = bytes / (10^6 × avg_time_ms)
```

### Q5: 候选内核列表是如何生成的？

**答**：在 `contractionCreatePlanPreference()` 时：

1. 获取所有预编译的内核模板：`allSolutions()`
2. 在 `contractionInitPlan()` 时按条件筛选：
   - 数据类型（typeA, typeB, typeD, computeType）
   - 操作类型（opA, opB）
3. 最终候选列表通常有 20-50 个内核
4. bruteForceModel 从中选择最优

### Q6: 首次运行为什么很慢？如何解决？

**答**：首次运行 `hiptensorCreatePlan()` 时需要遍历所有候选内核进行实际测试：

```
假设：
- 候选内核数量: 50 个
- 每个内核执行: 1(cold) + 1(hot) = 2 次
- 假设每次: 2ms
- 总耗时: 50 × 2 × 2ms = 200ms

解决方案：
- 相同参数的后续运行会命中 PlanCache，耗时约 1ms
- 加速比可达 200 倍
```

### Q7: PlanCache 缓存的是什么？

**答**：PlanCache 缓存的是**内核选择结果**（最优内核的 Uid），而不是编译后的代码。内核模板是预先编译好的，存储在 Composable Kernels (CK) 库中。PlanCache 记录的是：对于特定的问题参数（张量维度、数据类型等），应该使用哪个内核。

### Q8: α 和 β 参数是在什么时候赋值的？

**答**：α 和 β 在**执行时**（`hiptensorContract()` 等函数）才传入，而不是在创建操作描述符时。这样同一个 Plan 可以复用于不同的 α/β 值。

```cpp
// 执行时传入 α 和 β
hiptensorContract(handle, plan, &alpha, d_A, d_B, &beta,
                  d_C, d_D, workspace, workspaceSize, stream);
```

### Q9: 为什么收缩操作看起来"只有加法"？

**答**：收缩（Contraction）的数学定义是 Einstein 求和约定，核心就是对收缩维度求和。公式 `D = α*A*B + β*C` 中的加法是整体结果的累加，而非元素级操作。元素级的加法由 `HIPTENSOR_OP_ADD` 等一元/二元操作符提供。

### Q10: mPaddingLeft/mPaddingRight 如何用于多维张量？

**答**：这两个字段是为**最内层维度**的边界填充预留的 API，而不是整个张量的首尾填充。例如对于形状 (M, N) 的张量，填充作用于每行的首尾，而非所有数据之前/之后。

---

## 16. 相关资源

### 15.1 官方文档

- [hipTensor Documentation (ROCm)](https://rocm.docs.amd.com/projects/hipTensor/)
- [hipTensor API Reference](https://rocm.docs.amd.com/projects/hipTensor/en/latest/)
- [cuTensor Documentation (NVIDIA)](https://docs.nvidia.com/cuda/cutensor/)

### 15.2 源代码

- [ROCm/rocm-libraries GitHub](https://github.com/ROCm/rocm-libraries) - hipTensor 位于 `projects/hiptensor` 目录

### 15.3 依赖库

- **Composable Kernels (CK)**: hipTensor 的底层内核实现
- **HIP Runtime**: GPU 运行时支持

---

## 17. hipTensor 内核与 JIT 深度分析

### 17.1 内核存储机制

经过对 hipTensor 源码的深入分析，发现其内核管理机制如下：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    hipTensor 内核存储位置                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  编译时（Build Time）:                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  CK 模板 → 实例化多个内核变体 → 编译成机器码 → 打包进 .so         │   │
│  │                                                                 │   │
│  │  例如 Contraction 内核变体：                                      │   │
│  │  - FP32 + 列主序 + 大块策略 (BlockM=128, BlockN=128, BlockK=8)             │   │
│  │  - FP32 + 行主序 + 小块策略 (BlockM=64, BlockN=64, BlockK=16)              │   │
│  │  - FP16 + 列主序 + 大块策略                                       │   │
│  │  - ... 数十种组合                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  运行时（Runtime）:                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  GetInstances() → 从 .so 中加载预编译的内核                        │   │
│  │  ↓                                                              │   │
│  │  按条件过滤（数据类型、操作符、问题规模）                             │   │
│  │  ↓                                                              │   │
│  │  性能测试选出最优内核                                              │   │
│  │  ↓                                                              │   │
│  │  PlanCache 缓存选择结果（hash → uid）                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 17.2 JitMode 源码分析

**枚举定义**（`hiptensor_types.h`）：

```c
typedef enum {
    HIPTENSOR_JIT_MODE_NONE    = 0,  // 不使用 JIT
    HIPTENSOR_JIT_MODE_DEFAULT = 1,  // 使用 JIT（默认模式）
} hiptensorJitMode_t;
```

**实际使用情况**（`hiptensor.cpp`）：

```c
hiptensorStatus_t hiptensorCreatePlanPreference(...) {
    *pref = new hiptensorPlanPreference();
    // ...
    (*pref)->mJit = jitMode;  // ← 只是存储，从未被使用！
    // ...
}
```

**搜索整个源码树**，没有发现：
- `mJit` 的实际使用（判断或分支逻辑）
- hipRTC 调用
- 运行时编译相关代码

### 17.3 KernelCache 接口分析

```c
// hiptensor.cpp - 直接返回成功，空实现
hiptensorStatus_t hiptensorWriteKernelCacheToFile(...) {
    return HIPTENSOR_STATUS_SUCCESS;  // 什么都没做
}

hiptensorStatus_t hiptensorReadKernelCacheFromFile(...) {
    return HIPTENSOR_STATUS_SUCCESS;  // 什么都没做
}
```

### 17.4 结论汇总

| 功能 | hipTensor 状态 | 说明 |
|------|----------------|------|
| 预编译内核 | ✅ 完整实现 | 所有内核在编译时生成，存储在 .so 中 |
| GetInstances() | ✅ 完整实现 | 从预编译库中获取内核实例 |
| PlanCache | ✅ 完整实现 | 缓存选择结果（hash → uid） |
| **JitMode** | ❌ **未实现** | 接口存在，逻辑缺失 |
| **KernelCache 文件 I/O** | ❌ **空实现** | 接口存在，返回 SUCCESS |
| **运行时编译** | ❌ **不存在** | 无 hipRTC 调用 |

### 17.5 关键发现

**hipTensor 的 JIT 相关接口是"有接口无实现"的状态**：

1. **API 兼容性**：与 cuTENSOR 保持接口一致
2. **预留扩展**：为未来可能的功能预留
3. **当前状态**：所有内核都是预编译（AOT）的

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    hipTensor 内核管理真相                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  用户以为：                          实际情况：                           │
│  ┌─────────────────────────┐      ┌─────────────────────────────────┐  │
│  │ JitMode = DEFAULT       │  →  │ 所有内核都是预编译的             │  │
│  │ → 运行时 JIT 编译内核    │      │ → 从 .so 中加载                  │  │
│  │ → 针对问题优化           │      │ → 选择最合适的预编译内核          │  │
│  └─────────────────────────┘      └─────────────────────────────────┘  │
│                                                                         │
│  JitMode 参数被存储但从未被使用                                         │
│  KernelCache 接口存在但实现为空                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 附录A：Elementwise Binary vs Trinary 详细区别

> 本章节基于 hiptensor 源码分析，详细说明 Elementwise Binary 和 Trinary 的区别。

### A.1 数学公式

| 类型 | 公式 | 输入张量数 |
|------|------|-----------|
| **Binary** | `D = opAC(α·opA(A), γ·opC(C))` | 2 (A, C) |
| **Trinary** | `D = opABC(opAB(α·opA(A), β·opB(B)), γ·opC(C))` | 3 (A, B, C) |

### A.2 参数对比表

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Elementwise Binary vs Trinary                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BINARY: D = opAC(α·opA(A), γ·opC(C))                                       │
│  ─────────────────────────────────────                                      │
│  输入张量: A, C                                                              │
│  输出张量: D                                                                 │
│  标量系数: α, γ                                                              │
│  操作符: opA (一元), opC (一元), opAC (二元)                                  │
│  FLOPs: ~5 × problem_size                                                   │
│  数据量: 3 × problem_size × element_size (A + C + D)                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRINARY: D = opABC(opAB(α·opA(A), β·opB(B)), γ·opC(C))                     │
│  ───────────────────────────────────────────────────────────────────        │
│  输入张量: A, B, C                                                           │
│  输出张量: D                                                                 │
│  标量系数: α, β, γ                                                           │
│  操作符: opA (一元), opB (一元), opC (一元), opAB (二元), opABC (二元)        │
│  FLOPs: ~8 × problem_size                                                   │
│  数据量: 4 × problem_size × element_size (A + B + C + D)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### A.3 计算流程可视化

```
Binary 计算流程:
┌─────────┐    ┌─────────┐
│    A    │    │    C    │
└────┬────┘    └────┬────┘
     │              │
     ▼              ▼
┌─────────┐    ┌─────────┐
│  opA    │    │  opC    │       一元操作 (如 IDENTITY, SQRT, RELU)
└────┬────┘    └────┬────┘
     │              │
     ▼              ▼
┌─────────┐    ┌─────────┐
│  ×α     │    │  ×γ     │       标量缩放
└────┬────┘    └────┬────┘
     │              │
     └──────┬───────┘
            │
            ▼
      ┌───────────┐
      │   opAC    │              二元操作 (如 ADD, MUL, MAX, MIN)
      └─────┬─────┘
            │
            ▼
      ┌───────────┐
      │     D     │
      └───────────┘


Trinary 计算流程:
┌─────────┐    ┌─────────┐    ┌─────────┐
│    A    │    │    B    │    │    C    │
└────┬────┘    └────┬────┘    └────┬────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│  opA    │    │  opB    │    │  opC    │    一元操作
└────┬────┘    └────┬────┘    └────┬────┘
     │              │              │
     ▼              ▼              ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│  ×α     │    │  ×β     │    │  ×γ     │    标量缩放
└────┬────┘    └────┬────┘    └────┬────┘
     │              │              │
     └──────┬───────┘              │
            │                      │
            ▼                      │
      ┌───────────┐                │
      │   opAB    │                │         第一阶段二元操作
      └─────┬─────┘                │
            │                      │
            └──────────┬───────────┘
                       │
                       ▼
                 ┌───────────┐
                 │   opABC   │              第二阶段二元操作
                 └─────┬─────┘
                       │
                       ▼
                 ┌───────────┐
                 │     D     │
                 └───────────┘
```

### A.4 API 函数签名对比

```cpp
// Binary - 2个输入张量, 2个标量, 1个二元操作符
hiptensorStatus_t hiptensorElementwiseBinaryExecute(
    const hiptensorHandle_t handle,
    const hiptensorPlan_t   plan,
    const void*             alpha,    // A 的缩放系数
    const void*             A,        // 输入张量 A
    const void*             gamma,    // C 的缩放系数
    const void*             C,        // 输入张量 C
    void*                   D,        // 输出张量 D
    hipStream_t             stream);

// Trinary - 3个输入张量, 3个标量, 2个二元操作符
hiptensorStatus_t hiptensorElementwiseTrinaryExecute(
    const hiptensorHandle_t handle,
    const hiptensorPlan_t   plan,
    const void*             alpha,    // A 的缩放系数
    const void*             A,        // 输入张量 A
    const void*             beta,     // B 的缩放系数  ← Binary 没有
    const void*             B,        // 输入张量 B    ← Binary 没有
    const void*             gamma,    // C 的缩放系数
    const void*             C,        // 输入张量 C
    void*                   D,        // 输出张量 D
    hipStream_t             stream);
```

### A.5 操作描述符差异

```cpp
// Binary 操作描述符字段
struct BinaryOpDesc {
    TensorDescriptor descA, descC, descD;       // 3个张量描述符
    int32_t* modeA, modeC, modeD;               // 3个模式数组
    Operator opA, opC, opAC;                    // 1个二元 + 2个一元操作符
};

// Trinary 操作描述符字段
struct TrinaryOpDesc {
    TensorDescriptor descA, descB, descC, descD;  // 4个张量描述符 (多了 descB)
    int32_t* modeA, modeB, modeC, modeD;          // 4个模式数组 (多了 modeB)
    Operator opA, opB, opC, opAB, opABC;          // 2个二元 + 3个一元操作符
};
```

### A.6 hiptensor 源码中的查询参数对比

```cpp
// Binary 查询 - hiptensor_elementwise_binary.cpp:158-166
solutions = instances->query(
    {alphaF, gammaF},                              // 2个标量
    descA->mLengths,                               // A 的形状
    {descA->mType, descC->mType},                  // 2个输入类型
    {descD->mType},                                // 1个输出类型
    {{modeA}, {modeC}},                            // 2个输入模式
    {{modeD}},                                     // 1个输出模式
    {opAC, opA, opC},                              // 1个二元 + 2个一元操作符
    ExecutionSpace::DEVICE);

// Trinary 查询 - hiptensor_elementwise_trinary.cpp:175-185
solutions = instances->query(
    {alphaF, betaF, gammaF},                       // 3个标量
    descA->mLengths,                               // A 的形状
    {descA->mType, descB->mType, descC->mType},    // 3个输入类型
    {descD->mType},                                // 1个输出类型
    {{modeA}, {modeB}, {modeC}},                   // 3个输入模式
    {{modeD}},                                     // 1个输出模式
    {opABC, opAB, opA, opB, opC},                  // 2个二元 + 3个一元操作符
    ExecutionSpace::DEVICE);
```

### A.7 FLOPs 和内存带宽计算

```cpp
// Binary: ~5 ops/element - hiptensor_elementwise_binary.cpp:205
// 组成: 2次缩放(×α, ×γ) + 1次二元操作 + 开销
auto flops = std::size_t(5) * pSolution->problemSize();

// Trinary: ~8 ops/element - hiptensor_elementwise_trinary.cpp:225
// 组成: 3次缩放(×α, ×β, ×γ) + 2次二元操作 + 开销
auto flops = std::size_t(8) * pSolution->problemSize();

// Binary 内存传输量
auto bytes = (sizeof(A) + sizeof(C) + sizeof(D)) * problemSize();  // 3个张量

// Trinary 内存传输量
auto bytes = (sizeof(A) + sizeof(B) + sizeof(C) + sizeof(D)) * problemSize();  // 4个张量
```

### A.8 典型使用场景

| 类型 | 典型场景 | 示例 |
|------|---------|------|
| **Binary** | 两张量运算 | `D = A + C`, `D = A * C`, `D = max(A, C)` |
| **Trinary** | 融合运算 | `D = (A * B) + C` (融合乘加), `D = where(A > B, C, 0)` |

### A.9 Trinary 的性能优势

**为什么使用 Trinary？** 将两个连续的 Binary 操作融合为一次 Trinary 操作，减少全局内存读写。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    内存访问对比：两次 Binary vs 一次 Trinary                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  方案1: 两次 Binary 操作                                                    │
│  ─────────────────────                                                      │
│  D1 = A * B        读: A, B    写: D1     → 3次内存访问                     │
│  D  = D1 + C       读: D1, C   写: D     → 3次内存访问                     │
│  ───────────────────────────────────────────────────────────────────────    │
│  总计: 6次内存访问                                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  方案2: 一次 Trinary 操作                                                   │
│  ─────────────────────                                                      │
│  D = (A * B) + C   读: A, B, C  写: D    → 4次内存访问                     │
│  ───────────────────────────────────────────────────────────────────────    │
│  总计: 4次内存访问                                                          │
│                                                                             │
│  性能提升: 节省 33% 内存带宽                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### A.10 hiptensor 中的实例注册结构

```cpp
// hiptensor 按【维度 × 数据类型】分别注册 Binary 和 Trinary 解决方案

class ElementwiseSolutionInstances {
    // Binary 解决方案实例
    void ElementwiseBinarySolution2DFloatInstances();   // 2D FP32
    void ElementwiseBinarySolution2DHalfInstances();    // 2D FP16
    void ElementwiseBinarySolution3DFloatInstances();   // 3D FP32
    void ElementwiseBinarySolution3DHalfInstances();    // 3D FP16
    void ElementwiseBinarySolution4DFloatInstances();   // 4D FP32
    void ElementwiseBinarySolution4DHalfInstances();    // 4D FP16
    void ElementwiseBinarySolution5DFloatInstances();   // 5D FP32
    void ElementwiseBinarySolution5DHalfInstances();    // 5D FP16
    void ElementwiseBinarySolution6DFloatInstances();   // 6D FP32
    void ElementwiseBinarySolution6DHalfInstances();    // 6D FP16
    void ElementwiseBinarySolution2DDoubleInstances();  // 2D FP64
    void ElementwiseBinarySolution3DDoubleInstances();  // 3D FP64
    // ... 4D, 5D, 6D Double ...

    // Trinary 解决方案实例
    void ElementwiseTrinarySolution2DFloatInstances();  // 2D FP32
    void ElementwiseTrinarySolution2DHalfInstances();   // 2D FP16
    void ElementwiseTrinarySolution3DFloatInstances();  // 3D FP32
    void ElementwiseTrinarySolution3DHalfInstances();   // 3D FP16
    void ElementwiseTrinarySolution4DFloatInstances();  // 4D FP32
    void ElementwiseTrinarySolution4DHalfInstances();   // 4D FP16
    void ElementwiseTrinarySolution5DFloatInstances();  // 5D FP32
    void ElementwiseTrinarySolution5DHalfInstances();   // 5D FP16
    void ElementwiseTrinarySolution6DFloatInstances();  // 6D FP32
    void ElementwiseTrinarySolution6DHalfInstances();   // 6D FP16
    void ElementwiseTrinarySolution2DDoubleInstances(); // 2D FP64
    void ElementwiseTrinarySolution3DDoubleInstances(); // 3D FP64
    // ... 4D, 5D, 6D Double ...
};
```

### A.11 元素级操作符支持情况

| 操作符 | Binary (opAC) | Trinary (opAB/opABC) | 说明 |
|--------|---------------|---------------------|------|
| `HIPTENSOR_OP_ADD` | ✅ | ✅ | 加法 |
| `HIPTENSOR_OP_MUL` | ✅ | ✅ | 乘法 |
| `HIPTENSOR_OP_MAX` | ✅ | ✅ | 最大值 |
| `HIPTENSOR_OP_MIN` | ✅ | ✅ | 最小值 |
| `HIPTENSOR_OP_IDENTITY` | ✅ (opA/opC) | ✅ (opA/opB/opC) | 恒等（一元） |

### A.12 数据类型支持矩阵

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Binary 数据类型支持                                │
├────────────────────────────────────────────────────────────────────────────┤
│  {typeA, typeC, typeScalar}                                               │
│  ─────────────────────────────                                            │
│  {HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F}  ✅                 │
│  {HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_32F}  ✅                 │
│  {HIPTENSOR_R_16BF, HIPTENSOR_R_16BF, HIPTENSOR_R_16BF} ✅                │
│  {HIPTENSOR_R_16BF, HIPTENSOR_R_16BF, HIPTENSOR_R_32F} ✅                 │
│  {HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F}  ✅                 │
│  {HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F}  ✅                 │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                         Trinary 数据类型支持                               │
├────────────────────────────────────────────────────────────────────────────┤
│  {typeA, typeB, typeC, typeScalar}                                        │
│  ─────────────────────────────────────────                                │
│  {HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F} ✅  │
│  {HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F} ✅  │
│  {HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F} ✅  │
└────────────────────────────────────────────────────────────────────────────┘
```

### A.13 ops-tensor 实现建议

基于 hiptensor 的实现分析，ops-tensor 的 Elementwise Binary/Trinary 框架应该：

1. **共享基类 Solution**：Binary 和 Trinary 共用 `ElementwiseSolution` 基类
2. **分开的实例注册**：按维度(2D-6D) × 数据类型(FP16/FP32) 分别注册
3. **统一的查询接口**：通过输入张量数量区分 Binary(2个) / Trinary(3个)
4. **分开的 Execute 函数**：`acltensorElementwiseBinaryExecute` 和 `acltensorElementwiseTrinaryExecute`

```cpp
// 建议的 Solution 基类设计
class ElementwiseSolution {
public:
    virtual ~ElementwiseSolution() = default;

    // 统一的初始化接口
    virtual bool initArgs(
        std::vector<float> const& scalarValues,      // Binary: 2个, Trinary: 3个
        std::vector<std::vector<std::size_t>> const& inLengthsArray,
        std::vector<std::vector<std::size_t>> const& inStridesArray,
        std::vector<std::vector<int32_t>> const& inModesArray,
        std::vector<std::vector<std::size_t>> const& outLengthsArray,
        std::vector<std::vector<std::size_t>> const& outStridesArray,
        std::vector<std::vector<int32_t>> const& outModesArray,
        std::vector<acltensorOperator_t> const& operators,  // Binary: 3个, Trinary: 5个
        std::vector<const void*> const& inBuffers,         // Binary: 2个, Trinary: 3个
        std::vector<void*> const& outBuffers) = 0;

    // 执行
    virtual float operator()(StreamConfig const& config) = 0;

    // 属性
    virtual uint64_t uid() const = 0;
    virtual size_t problemSize() const = 0;
    virtual const char* kernelName() const = 0;
};
```

---

**文档版本**：基于 ROCm hipTensor (rocm-libraries develop 分支)
**最后更新**：2026-03-14
