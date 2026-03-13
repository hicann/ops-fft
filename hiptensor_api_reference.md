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

**文档版本**：基于 ROCm hipTensor (rocm-libraries develop 分支)
**最后更新**：2026-03-13
