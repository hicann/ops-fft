# CANN ops-tensor 阶段一实施方案：Elementwise 能力建设

> 基于 `ops_tensor_api_design.md` 设计方案，结合 hiptensor 实现参考
>
> 文档创建时间：2024年

---

## 一、总体策略

### 1.1 阶段一目标

**最小可行版本**：实现 Elementwise Binary 核心框架，支持 **Add** 算子，
仅支持 **FP32** 数据类型

### 1.2 实施原则

| 原则 | 说明 |
|------|------|
| **最小化优先** | 先实现最核心的功能路径，跑通端到端流程 |
| **预留标记** | 非必要接口/字段用 `// TODO: Phase 2` 注释预留 |
| **渐进扩展** | 阶段一只支持 FP32，后续扩展 FP16/BF16 等 |

---

## 二、接口/字段裁剪方案

### 2.1 阶段一 **需要实现** 的内容

| 类别 | 接口/字段 | 说明 |
|------|----------|------|
| **句柄管理** | `acltensorCreate`, `acltensorDestroy` | 核心 |
| **张量描述符** | `acltensorCreateTensorDescriptor`, `acltensorDestroyTensorDescriptor` | 核心 |
| **操作描述符** | `acltensorCreateElementwiseBinary` | **阶段一重点（仅 Binary）** |
| **Plan管理** | `acltensorCreatePlanPreference`, `acltensorCreatePlan`, `acltensorDestroyPlan` | 核心 |
| **执行函数** | `acltensorElementwiseBinaryExecute` | **阶段一重点（仅 Binary）** |
| **操作符** | `ADD` + `IDENTITY` | 阶段一只支持 ADD |
| **数据类型** | `ACLTENSOR_R_32F` | 阶段一只支持 FP32 |
| **辅助函数** | `acltensorGetErrorString`, `acltensorGetVersion` | 基础工具 |

### 2.2 阶段一 **标记预留** 的内容

| 类别 | 接口/字段 | 预留原因 | 阶段 |
|------|----------|----------|------|
| **句柄管理** | `acltensorHandleResizePlanCache` | PlanCache 后期实现 | Phase 2 |
| | `acltensorHandleWritePlanCacheToFile` | 文件持久化后期实现 | Phase 2 |
| | `acltensorHandleReadPlanCacheFromFile` | 文件持久化后期实现 | Phase 2 |
| **操作描述符** | `acltensorCreateContraction` | 收缩操作 | Phase 2 |
| | `acltensorCreateReduction` | 归约操作 | Phase 3 |
| | `acltensorCreatePermutation` | 置换操作 | Phase 4 |
| | `acltensorCreateElementwiseTrinary` | 三元逐元素 | Phase 2 |
| | `acltensorOperationDescriptorSetAttribute` | 属性设置 | Phase 2 |
| | `acltensorOperationDescriptorGetAttribute` | 属性获取 | Phase 2 |
| **执行函数** | `acltensorElementwiseTrinaryExecute` | 三元逐元素执行 | Phase 2 |
| | `acltensorContract` | 收缩执行 | Phase 2 |
| | `acltensorReduce` | 归约执行 | Phase 3 |
| | `acltensorPermute` | 置换执行 | Phase 4 |
| **操作符 - 二元** | `MUL`, `SUB`, `DIV`, `MAX`, `MIN` | 基础二元操作符扩展 | Phase 2 |
| | `POW` | 复杂二元操作 | Phase 3 |
| **操作符 - 一元** | `SQRT/RELU/CONJ/RCP/SIGMOID/TANH` | 一元操作符 | Phase 2 |
| | `EXP/LOG/ABS/NEG/SIN/COS/TAN` | 一元操作符 | Phase 2 |
| | `SINH/COSH/ASIN/ACOS/ATAN/CEIL/FLOOR` | 一元操作符 | Phase 2 |
| | `GELU/SILU` | 激活函数 | Phase 3 |
| **数据类型** | `ACLTENSOR_R_16F` | FP16 | Phase 2 |
| | `ACLTENSOR_R_16BF` | BF16 | Phase 2 |
| | `ACLTENSOR_R_8I`, `ACLTENSOR_R_8U` | INT8 | Phase 3 |
| | `ACLTENSOR_R_32I`, `ACLTENSOR_R_32U` | INT32 | Phase 3 |
| | `ACLTENSOR_R_64F` | FP64 | Phase 3 |
| | `ACLTENSOR_C_32F`, `ACLTENSOR_C_64F` | 复数 | Phase 4 |
| **Plan属性** | `ACLTENSOR_PLAN_KERNEL_ID` | 内核ID | Phase 2 |
| **PlanPreference属性** | `ACLTENSOR_PLAN_PREFERENCE_CACHE_MODE` | 缓存模式 | Phase 2 |
| | `ACLTENSOR_PLAN_PREFERENCE_KERNEL_RANK` | Contraction专用 | Phase 2 |
| **工作空间偏好** | `ACLTENSOR_WORKSPACE_MIN/MAX` | Elementwise不需要workspace | 预留 |
| **日志系统** | 全部 6 个接口 | 日志功能 | Phase 2 |
| **PlanCache** | 完整缓存机制 | 内核选择缓存 | Phase 2 |

### 2.3 接口状态汇总表

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          接口实现状态（阶段一）                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  句柄管理 (5个)                                                              │
│  ├── ✓ acltensorCreate                        [阶段一实现]                   │
│  ├── ✓ acltensorDestroy                       [阶段一实现]                   │
│  ├── ○ acltensorHandleResizePlanCache         [预留 Phase 2]                │
│  ├── ○ acltensorHandleWritePlanCacheToFile    [预留 Phase 2]                │
│  └── ○ acltensorHandleReadPlanCacheFromFile   [预留 Phase 2]                │
│                                                                             │
│  张量描述符 (2个)                                                            │
│  ├── ✓ acltensorCreateTensorDescriptor        [阶段一实现-仅FP32]            │
│  └── ✓ acltensorDestroyTensorDescriptor       [阶段一实现]                   │
│                                                                             │
│  操作描述符 (9个)                                                            │
│  ├── ✓ acltensorCreateElementwiseBinary       [阶段一实现]                   │
│  ├── ○ acltensorCreateElementwiseTrinary      [预留 Phase 2]   ← 调整       │
│  ├── ✓ acltensorDestroyOperationDescriptor    [阶段一实现]                   │
│  ├── ○ acltensorCreateContraction             [预留 Phase 3]   ← 调整       │
│  ├── ○ acltensorCreateReduction               [预留 Phase 4]   ← 调整       │
│  ├── ○ acltensorCreatePermutation             [预留 Phase 5]   ← 调整       │
│  ├── ○ acltensorOperationDescriptorSetAttribute [预留 Phase 2]              │
│  └── ○ acltensorOperationDescriptorGetAttribute [预留 Phase 2]              │
│                                                                             │
│  Plan管理 (7个)                                                              │
│  ├── ✓ acltensorCreatePlanPreference          [阶段一实现-简化版]            │
│  ├── ✓ acltensorDestroyPlanPreference         [阶段一实现]                   │
│  ├── ✓ acltensorCreatePlan                    [阶段一实现]                   │
│  ├── ✓ acltensorDestroyPlan                   [阶段一实现]                   │
│  ├── ○ acltensorPlanPreferenceSetAttribute    [预留 Phase 2]                │
│  ├── ○ acltensorPlanGetAttribute              [预留 Phase 2]                │
│  └── ○ acltensorEstimateWorkspaceSize         [预留 Phase 2]                │
│                                                                             │
│  执行函数 (5个)                                                              │
│  ├── ✓ acltensorElementwiseBinaryExecute      [阶段一实现-仅ADD+FP32]        │
│  ├── ○ acltensorElementwiseTrinaryExecute     [预留 Phase 2]   ← 调整       │
│  ├── ○ acltensorContract                      [预留 Phase 3]   ← 调整       │
│  ├── ○ acltensorReduce                        [预留 Phase 4]   ← 调整       │
│  └── ○ acltensorPermute                       [预留 Phase 5]   ← 调整       │
│                                                                             │
│  辅助工具 (2个)                                                              │
│  ├── ✓ acltensorGetErrorString                [阶段一实现]                   │
│  └── ✓ acltensorGetVersion                    [阶段一实现]                   │
│                                                                             │
│  日志系统 (6个)                                                              │
│  ├── ○ acltensorLoggerSetCallback             [预留 Phase 2]                │
│  ├── ○ acltensorLoggerSetFile                 [预留 Phase 2]                │
│  ├── ○ acltensorLoggerOpenFile                [预留 Phase 2]                │
│  ├── ○ acltensorLoggerSetLevel                [预留 Phase 2]                │
│  ├── ○ acltensorLoggerSetMask                 [预留 Phase 2]                │
│  └── ○ acltensorLoggerForceDisable            [预留 Phase 2]                │
│                                                                             │
│  图例: ✓ 实现中  ○ 预留                                                      │
│                                                                             │
│  阶段一核心范围: Elementwise Binary + ADD算子 + FP32                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、阶段一代码结构

```
ops-tensor/
├── include/                              # 公共头文件
│   ├── cann_ops_tensor.h                 # 主 API（阶段一精简版）
│   └── cann_ops_tensor_types.h           # 类型定义（阶段一精简版）
│
├── api/                                  # 上层框架实现
│   ├── core/                             # 核心框架 ✓ 阶段一
│   │   ├── handle.cpp                    # 句柄管理（简化版，无 PlanCache）
│   │   ├── handle.hpp
│   │   ├── handle_impl.hpp               # 句柄内部实现
│   │   ├── tensor_descriptor.cpp         # 张量描述符（仅 FP32）
│   │   ├── tensor_descriptor.hpp
│   │   ├── operation_descriptor.cpp      # 操作描述符（仅 Binary）
│   │   ├── operation_descriptor.hpp
│   │   ├── plan.cpp                      # Plan 管理
│   │   ├── plan.hpp
│   │   ├── plan_preference.cpp           # Plan 偏好（简化版）
│   │   ├── plan_preference.hpp
│   │   └── device_info.hpp               # 设备信息查询
│   │
│   ├── elementwise/                      # Elementwise 框架 ✓ 阶段一重点
│   │   ├── elementwise_binary.cpp        # Binary 分发逻辑
│   │   ├── elementwise_binary.hpp
│   │   ├── elementwise_solution.hpp      # 解决方案抽象
│   │   ├── elementwise_solution_impl.hpp # 解决方案实现
│   │   └── elementwise_solution_registry.hpp  # 内核注册表
│   │
│   └── utils/                            # 工具函数
│       ├── tiling_utils.hpp              # Tiling 计算工具
│       ├── type_utils.hpp                # 类型转换工具（仅 FP32）
│       └── error_utils.hpp               # 错误处理工具
│
├── src/                                  # 算子 Kernel 实现 ✓ 阶段一
│   │
│   └── add/                              # Add 算子（阶段一唯一算子）
│       ├── arch35/
│       │   └── add_tiling.h              # Tiling 数据结构
│       ├── add_host.cpp                  # 算子 Host 端（仅 FP32）
│       ├── add_host.h
│       └── add_kernel.cpp                # AscendC Kernel 实现（仅 FP32）
│
├── tests/                                # 测试代码
│   ├── CMakeLists.txt
│   ├── test_common.h                     # 测试公共头文件
│   └── test_elementwise_binary_add.cpp   # Elementwise Binary Add 端到端测试
│
├── CMakeLists.txt                        # 主构建脚本
└── README.md                             # 项目说明
```

---

## 四、头文件设计（阶段一精简版）

### 4.1 `include/cann_ops_tensor_types.h`

```c
#ifndef CANN_OPS_TENSOR_TYPES_H
#define CANN_OPS_TENSOR_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * 1. 状态码定义
 *============================================================================*/
typedef enum {
    ACLTENSOR_STATUS_SUCCESS = 0,
    ACLTENSOR_STATUS_NOT_INITIALIZED = 1,
    ACLTENSOR_STATUS_ALLOC_FAILED = 3,
    ACLTENSOR_STATUS_INVALID_VALUE = 7,
    ACLTENSOR_STATUS_ARCH_MISMATCH = 8,
    ACLTENSOR_STATUS_EXECUTION_FAILED = 13,
    ACLTENSOR_STATUS_INTERNAL_ERROR = 14,
    ACLTENSOR_STATUS_NOT_SUPPORTED = 15,
    ACLTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    ACLTENSOR_STATUS_INSUFFICIENT_DRIVER = 20,
    ACLTENSOR_STATUS_IO_ERROR = 21,
} acltensorStatus_t;

/*============================================================================
 * 2. 数据类型枚举
 *============================================================================*/
typedef enum {
    ACLTENSOR_R_16F = 0,     /* FP16 半精度浮点 - 阶段一支持 */
    ACLTENSOR_R_32F = 1,     /* FP32 单精度浮点 - 阶段一支持 */

    /* TODO: Phase 2 - 以下数据类型阶段一预留 */
    ACLTENSOR_R_64F = 2,     /* FP64 双精度浮点 */
    ACLTENSOR_R_16BF = 3,    /* BF16 BFloat16 */
    ACLTENSOR_R_8I = 4,      /* INT8 有符号8位整数 */
    ACLTENSOR_R_8U = 5,      /* UINT8 无符号8位整数 */
    ACLTENSOR_R_32I = 6,     /* INT32 有符号32位整数 */
    ACLTENSOR_R_32U = 7,     /* UINT32 无符号32位整数 */
    ACLTENSOR_C_32F = 8,     /* Complex FP32 */
    ACLTENSOR_C_64F = 9,     /* Complex FP64 */
} acltensorDataType_t;

/*============================================================================
 * 3. 计算精度描述符
 *============================================================================*/
typedef enum {
    ACLTENSOR_COMPUTE_DESC_16F = (1U << 0),   /* FP16 计算 - 阶段一支持 */
    ACLTENSOR_COMPUTE_DESC_32F = (1U << 2),   /* FP32 计算 - 阶段一支持 */
    ACLTENSOR_COMPUTE_DESC_NONE = 0,

    /* TODO: Phase 2 - 以下计算精度阶段一预留 */
    ACLTENSOR_COMPUTE_DESC_64F = (1U << 4),
    ACLTENSOR_COMPUTE_DESC_16BF = (1U << 10),
    ACLTENSOR_COMPUTE_DESC_C32F = (1U << 11),
    ACLTENSOR_COMPUTE_DESC_C64F = (1U << 12),
} acltensorComputeDescriptor_t;

/*============================================================================
 * 4. 操作符枚举
 *============================================================================*/
typedef enum {
    /* 一元操作符 - 阶段一仅支持 IDENTITY */
    ACLTENSOR_OP_IDENTITY = 1,  /* 恒等: y = x - 阶段一支持 */

    /* 二元操作符 - 阶段一全部支持 */
    ACLTENSOR_OP_ADD = 3,       /* 加法: y = a + b */
    ACLTENSOR_OP_MUL = 5,       /* 乘法: y = a * b */
    ACLTENSOR_OP_MAX = 6,       /* 最大值: y = max(a, b) */
    ACLTENSOR_OP_MIN = 7,       /* 最小值: y = min(a, b) */

    /* TODO: Phase 2 - 以下一元操作符预留 */
    ACLTENSOR_OP_SQRT = 2,      /* 平方根 */
    ACLTENSOR_OP_RELU = 8,      /* ReLU */
    ACLTENSOR_OP_CONJ = 9,      /* 共轭 */
    ACLTENSOR_OP_RCP = 10,      /* 倒数 */
    ACLTENSOR_OP_SIGMOID = 11,  /* Sigmoid */
    ACLTENSOR_OP_TANH = 12,     /* Tanh */
    ACLTENSOR_OP_EXP = 22,      /* 指数 */
    ACLTENSOR_OP_LOG = 23,      /* 对数 */
    ACLTENSOR_OP_ABS = 24,      /* 绝对值 */
    ACLTENSOR_OP_NEG = 25,      /* 取负 */
    ACLTENSOR_OP_SIN = 26,      /* 正弦 */
    ACLTENSOR_OP_COS = 27,      /* 余弦 */
    ACLTENSOR_OP_TAN = 28,      /* 正切 */
    ACLTENSOR_OP_SINH = 29,     /* 双曲正弦 */
    ACLTENSOR_OP_COSH = 30,     /* 双曲余弦 */
    ACLTENSOR_OP_ASIN = 31,     /* 反正弦 */
    ACLTENSOR_OP_ACOS = 32,     /* 反余弦 */
    ACLTENSOR_OP_ATAN = 33,     /* 反正切 */
    ACLTENSOR_OP_CEIL = 37,     /* 向上取整 */
    ACLTENSOR_OP_FLOOR = 38,    /* 向下取整 */
    ACLTENSOR_OP_GELU = 39,     /* GELU 激活函数 */
    ACLTENSOR_OP_SILU = 40,     /* SiLU/Swish 激活函数 */

    /* TODO: Phase 2 - 以下二元操作符预留 */
    ACLTENSOR_OP_SUB = 50,      /* 减法: y = a - b */
    ACLTENSOR_OP_DIV = 51,      /* 除法: y = a / b */
    ACLTENSOR_OP_POW = 52,      /* 幂运算: y = a^b */

    ACLTENSOR_OP_UNKNOWN = 126,
} acltensorOperator_t;

/*============================================================================
 * 5. 算法选择
 *============================================================================*/
typedef enum {
    ACLTENSOR_ALGO_DEFAULT = -1,          /* 启发式自动选择 - 阶段一支持 */

    /* TODO: Phase 2 */
    ACLTENSOR_ALGO_DEFAULT_PATIENT = -6,  /* 更精确但耗时的选择 */
} acltensorAlgo_t;

/*============================================================================
 * 6. 工作空间偏好
 *============================================================================*/
typedef enum {
    ACLTENSOR_WORKSPACE_MIN = 1,
    ACLTENSOR_WORKSPACE_DEFAULT = 2,      /* 阶段一支持 */
    ACLTENSOR_WORKSPACE_MAX = 3,
} acltensorWorksizePreference_t;

/*============================================================================
 * 7. 缓存模式 - TODO: Phase 2
 *============================================================================*/
typedef enum {
    ACLTENSOR_CACHE_MODE_NONE = 0,
    ACLTENSOR_CACHE_MODE_PEDANTIC = 1,
} acltensorCacheMode_t;

/*============================================================================
 * 8. 日志级别 - TODO: Phase 2
 *============================================================================*/
typedef enum {
    ACLTENSOR_LOG_LEVEL_OFF = 0,
    ACLTENSOR_LOG_LEVEL_ERROR = 1,
    ACLTENSOR_LOG_LEVEL_PERF_TRACE = 2,
    ACLTENSOR_LOG_LEVEL_PERF_HINT = 4,
    ACLTENSOR_LOG_LEVEL_HEURISTICS_TRACE = 8,
    ACLTENSOR_LOG_LEVEL_API_TRACE = 16,
} acltensorLogLevel_t;

/*============================================================================
 * 9. 操作描述符属性 - TODO: Phase 2
 *============================================================================*/
typedef enum {
    ACLTENSOR_OPERATION_DESCRIPTOR_TAG = 0,
    ACLTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE = 1,
    ACLTENSOR_OPERATION_DESCRIPTOR_FLOPS = 2,
    ACLTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES = 3,
    ACLTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT = 4,
    ACLTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT = 5,
} acltensorOperationDescriptorAttribute_t;

/*============================================================================
 * 10. Plan 属性 - TODO: Phase 2
 *============================================================================*/
typedef enum {
    ACLTENSOR_PLAN_REQUIRED_WORKSPACE = 0,
    ACLTENSOR_PLAN_KERNEL_ID = 1,
} acltensorPlanAttribute_t;

/*============================================================================
 * 11. PlanPreference 属性 - TODO: Phase 2
 *============================================================================*/
typedef enum {
    ACLTENSOR_PLAN_PREFERENCE_CACHE_MODE = 0,
    ACLTENSOR_PLAN_PREFERENCE_ALGO = 1,
    ACLTENSOR_PLAN_PREFERENCE_KERNEL_RANK = 2,
} acltensorPlanPreferenceAttribute_t;

/*============================================================================
 * 12. 不透明句柄类型
 *============================================================================*/
typedef struct acltensorHandle* acltensorHandle_t;
typedef struct acltensorTensorDescriptor* acltensorTensorDescriptor_t;
typedef struct acltensorOperationDescriptor* acltensorOperationDescriptor_t;
typedef struct acltensorPlanPreference* acltensorPlanPreference_t;
typedef struct acltensorPlan* acltensorPlan_t;

/*============================================================================
 * 13. 日志回调函数类型 - TODO: Phase 2
 *============================================================================*/
typedef void (*acltensorLoggerCallback_t)(int32_t logContext,
                                          const char* funcName,
                                          const char* msg);

#ifdef __cplusplus
}
#endif

#endif /* CANN_OPS_TENSOR_TYPES_H */
```

### 4.2 `include/cann_ops_tensor.h`

```c
#ifndef CANN_OPS_TENSOR_H
#define CANN_OPS_TENSOR_H

#include "cann_ops_tensor_types.h"
#include "acl/acl.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 *                        1. 句柄管理 (Handle Management)
 *============================================================================*/

/**
 * @brief 创建并初始化 ops-tensor 库句柄
 * @param[out] handle  返回的句柄指针
 * @return ACLTENSOR_STATUS_SUCCESS 成功，否则返回错误码
 */
acltensorStatus_t acltensorCreate(acltensorHandle_t* handle);

/**
 * @brief 销毁 ops-tensor 库句柄
 * @param[in] handle  要销毁的句柄
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
acltensorStatus_t acltensorDestroy(acltensorHandle_t handle);

/* TODO: Phase 2 - PlanCache 相关接口 */
acltensorStatus_t acltensorHandleResizePlanCache(
    acltensorHandle_t handle, const uint32_t numEntries);

acltensorStatus_t acltensorHandleWritePlanCacheToFile(
    const acltensorHandle_t handle, const char fileName[]);

acltensorStatus_t acltensorHandleReadPlanCacheFromFile(
    acltensorHandle_t handle, const char fileName[], uint32_t* numCachelinesRead);

/*============================================================================
 *                        2. 张量描述符管理 (Tensor Descriptor)
 *============================================================================*/

acltensorStatus_t acltensorCreateTensorDescriptor(
    const acltensorHandle_t      handle,
    acltensorTensorDescriptor_t* desc,
    const uint32_t               numModes,
    const int64_t                lens[],
    const int64_t                strides[],
    acltensorDataType_t          dataType,
    uint32_t                     alignmentRequirement);

acltensorStatus_t acltensorDestroyTensorDescriptor(acltensorTensorDescriptor_t desc);

/*============================================================================
 *                        3. 操作描述符管理 (Operation Descriptor)
 *============================================================================*/

/* 阶段一重点实现 */
acltensorStatus_t acltensorCreateElementwiseBinary(
    const acltensorHandle_t            handle,
    acltensorOperationDescriptor_t*    desc,
    const acltensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    acltensorOperator_t                opA,
    const acltensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    acltensorOperator_t                opC,
    const acltensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    acltensorOperator_t                opAC,
    const acltensorComputeDescriptor_t descCompute);

acltensorStatus_t acltensorCreateElementwiseTrinary(
    const acltensorHandle_t            handle,
    acltensorOperationDescriptor_t*    desc,
    const acltensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    acltensorOperator_t                opA,
    const acltensorTensorDescriptor_t  descB,
    const int32_t                      modeB[],
    acltensorOperator_t                opB,
    const acltensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    acltensorOperator_t                opC,
    const acltensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    acltensorOperator_t                opAB,
    acltensorOperator_t                opABC,
    const acltensorComputeDescriptor_t descCompute);

acltensorStatus_t acltensorDestroyOperationDescriptor(acltensorOperationDescriptor_t desc);

/* TODO: Phase 2 - 其他操作类型 */
acltensorStatus_t acltensorCreateContraction(
    const acltensorHandle_t            handle,
    acltensorOperationDescriptor_t*    desc,
    const acltensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    acltensorOperator_t                opA,
    const acltensorTensorDescriptor_t  descB,
    const int32_t                      modeB[],
    acltensorOperator_t                opB,
    const acltensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    acltensorOperator_t                opC,
    const acltensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    const acltensorComputeDescriptor_t descCompute);

/* TODO: Phase 3 */
acltensorStatus_t acltensorCreateReduction(
    const acltensorHandle_t            handle,
    acltensorOperationDescriptor_t*    desc,
    const acltensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    acltensorOperator_t                opA,
    const acltensorTensorDescriptor_t  descC,
    const int32_t                      modeC[],
    acltensorOperator_t                opC,
    const acltensorTensorDescriptor_t  descD,
    const int32_t                      modeD[],
    acltensorOperator_t                opReduce,
    const acltensorComputeDescriptor_t descCompute);

/* TODO: Phase 4 */
acltensorStatus_t acltensorCreatePermutation(
    const acltensorHandle_t            handle,
    acltensorOperationDescriptor_t*    desc,
    const acltensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    acltensorOperator_t                opA,
    const acltensorTensorDescriptor_t  descB,
    const int32_t                      modeB[],
    const acltensorComputeDescriptor_t descCompute);

/* TODO: Phase 2 - 属性接口 */
acltensorStatus_t acltensorOperationDescriptorSetAttribute(
    const acltensorHandle_t                 handle,
    acltensorOperationDescriptor_t          desc,
    acltensorOperationDescriptorAttribute_t attr,
    const void*                             buf,
    size_t                                  sizeInBytes);

acltensorStatus_t acltensorOperationDescriptorGetAttribute(
    const acltensorHandle_t                 handle,
    acltensorOperationDescriptor_t          desc,
    acltensorOperationDescriptorAttribute_t attr,
    void*                                   buf,
    size_t                                  sizeInBytes);

/*============================================================================
 *                        4. Plan 和 PlanPreference 管理
 *============================================================================*/

/* 阶段一实现简化版 */
acltensorStatus_t acltensorCreatePlanPreference(
    const acltensorHandle_t    handle,
    acltensorPlanPreference_t* pref,
    acltensorAlgo_t            algo);

acltensorStatus_t acltensorDestroyPlanPreference(acltensorPlanPreference_t pref);

acltensorStatus_t acltensorCreatePlan(
    const acltensorHandle_t              handle,
    acltensorPlan_t*                     plan,
    const acltensorOperationDescriptor_t desc,
    const acltensorPlanPreference_t      pref,
    uint64_t                             workspaceSizeLimit);

acltensorStatus_t acltensorDestroyPlan(acltensorPlan_t plan);

/* TODO: Phase 2 */
acltensorStatus_t acltensorPlanPreferenceSetAttribute(
    const acltensorHandle_t            handle,
    acltensorPlanPreference_t          pref,
    acltensorPlanPreferenceAttribute_t attr,
    const void*                        buf,
    size_t                             sizeInBytes);

acltensorStatus_t acltensorEstimateWorkspaceSize(
    const acltensorHandle_t              handle,
    const acltensorOperationDescriptor_t desc,
    const acltensorPlanPreference_t      planPref,
    const acltensorWorksizePreference_t  workspacePref,
    uint64_t*                            workspaceSizeEstimate);

acltensorStatus_t acltensorPlanGetAttribute(
    const acltensorHandle_t  handle,
    const acltensorPlan_t    plan,
    acltensorPlanAttribute_t attr,
    void*                    buf,
    size_t                   sizeInBytes);

/*============================================================================
 *                        5. 执行函数 (Execution)
 *============================================================================*/

/* 阶段一重点实现 */
acltensorStatus_t acltensorElementwiseBinaryExecute(
    const acltensorHandle_t handle,
    const acltensorPlan_t   plan,
    const void*             alpha,
    const void*             A,
    const void*             gamma,
    const void*             C,
    void*                   D,
    aclrtStream             stream);

acltensorStatus_t acltensorElementwiseTrinaryExecute(
    const acltensorHandle_t handle,
    const acltensorPlan_t   plan,
    const void*             alpha,
    const void*             A,
    const void*             beta,
    const void*             B,
    const void*             gamma,
    const void*             C,
    void*                   D,
    aclrtStream             stream);

/* TODO: Phase 2 - Contraction */
acltensorStatus_t acltensorContract(
    const acltensorHandle_t handle,
    const acltensorPlan_t   plan,
    const void*             alpha,
    const void*             A,
    const void*             B,
    const void*             beta,
    const void*             C,
    void*                   D,
    void*                   workspace,
    uint64_t                workspaceSize,
    aclrtStream             stream);

/* TODO: Phase 3 - Reduction */
acltensorStatus_t acltensorReduce(
    const acltensorHandle_t handle,
    const acltensorPlan_t   plan,
    const void*             alpha,
    const void*             A,
    const void*             beta,
    const void*             C,
    void*                   D,
    void*                   workspace,
    uint64_t                workspaceSize,
    aclrtStream             stream);

/* TODO: Phase 4 - Permutation */
acltensorStatus_t acltensorPermute(
    const acltensorHandle_t handle,
    const acltensorPlan_t   plan,
    const void*             alpha,
    const void*             A,
    void*                   B,
    aclrtStream             stream);

/*============================================================================
 *                        6. 辅助工具函数 (Utilities)
 *============================================================================*/

const char* acltensorGetErrorString(const acltensorStatus_t error);
size_t acltensorGetVersion(void);

/*============================================================================
 *                        7. 日志系统 (Logging) - TODO: Phase 2
 *============================================================================*/

acltensorStatus_t acltensorLoggerSetCallback(acltensorLoggerCallback_t callback);
acltensorStatus_t acltensorLoggerSetFile(FILE* file);
acltensorStatus_t acltensorLoggerOpenFile(const char* logFile);
acltensorStatus_t acltensorLoggerSetLevel(acltensorLogLevel_t level);
acltensorStatus_t acltensorLoggerSetMask(int32_t mask);
acltensorStatus_t acltensorLoggerForceDisable(void);

#ifdef __cplusplus
}
#endif

#endif /* CANN_OPS_TENSOR_H */
```

---

## 五、核心结构体内部定义

### 5.1 Handle 结构体（阶段一简化版）

```cpp
// api/core/handle_impl.hpp

#include <mutex>
#include <memory>
#include "acl/acl.h"

namespace acltensor {

// Ascend 设备信息
class AscendDevice {
public:
    AscendDevice();
    ~AscendDevice() = default;

    int32_t  getDeviceId()   const { return mDeviceId; }
    uint32_t getCoreNum()    const { return mCoreNum; }    // AIV 核数
    uint64_t getUbSize()     const { return mUbSize; }     // UB 大小
    const char* getSoCName() const { return mSoCName; }

    // 能力查询 - 阶段一只关注 FP16/FP32
    bool supportsFp16() const { return mSupportsFp16; }
    bool supportsFp32() const { return true; }  // 始终支持

    /* TODO: Phase 2 */
    // bool supportsBf16() const { return mSupportsBf16; }
    // bool supportsInt8() const { return mSupportsInt8; }

private:
    int32_t  mDeviceId     = -1;
    uint32_t mCoreNum      = 0;
    uint64_t mUbSize       = 0;
    char     mSoCName[64]  = {0};
    bool     mSupportsFp16 = false;
};

/* TODO: Phase 2 - PlanCache
class PlanCache {
public:
    explicit PlanCache(uint32_t maxEntries = 64);
    uint64_t querySolutionUid(uint64_t hashId);
    void addCacheLine(uint64_t hashId, uint64_t solutionUid);
    void resize(uint32_t numEntries);
    bool writeFile(const char* fileName);
    bool readFile(const char* fileName, uint32_t* numRead);
private:
    uint32_t mMaxEntries;
    std::unordered_map<uint64_t, uint64_t> mCacheLines;
    std::mutex mMutex;
};
*/

} // namespace acltensor

// 对外句柄结构体（阶段一简化版）
struct acltensorHandle {
    acltensor::AscendDevice device;

    /* TODO: Phase 2 */
    // acltensor::PlanCache* planCache = nullptr;
    // acltensorLogLevel_t logLevel = ACLTENSOR_LOG_LEVEL_OFF;
};
```

### 5.2 TensorDescriptor 结构体

```cpp
// api/core/tensor_descriptor.hpp

#include <vector>
#include <cstdint>

struct acltensorTensorDescriptor {
    // 基本属性
    acltensorDataType_t  dataType;
    uint32_t             numModes;
    std::vector<int64_t> lens;      // 各维度长度
    std::vector<int64_t> strides;   // 各维度步长
    uint32_t             alignmentRequirement;

    // 派生属性（创建时计算）
    size_t               elementSize;    // 单元素字节数
    size_t               totalElements;  // 总元素数
    size_t               totalBytes;     // 总字节数
    bool                 isContiguous;   // 是否连续内存

    // 辅助方法
    size_t getElementSize(acltensorDataType_t type) const;
    void   computeDerivedAttributes();
    bool   checkContiguous() const;
};
```

### 5.3 OperationDescriptor 结构体（阶段一 Elementwise 专用）

```cpp
// api/core/operation_descriptor.hpp

#include <vector>

// 操作类型枚举
enum class OperationType : uint32_t {
    ELEMENTWISE_BINARY = 0,   // 阶段一
    ELEMENTWISE_TRINARY = 1,  // 阶段一
    /* TODO: Phase 2-4 */
    // CONTRACTION = 2,
    // REDUCTION = 3,
    // PERMUTATION = 4,
};

struct acltensorOperationDescriptor {
    // 操作类型
    OperationType operationType;

    // ========== 张量描述符 ==========
    acltensorTensorDescriptor_t descA = nullptr;
    acltensorTensorDescriptor_t descB = nullptr;  // 仅 Trinary 使用
    acltensorTensorDescriptor_t descC = nullptr;
    acltensorTensorDescriptor_t descD = nullptr;

    // ========== 模式数组（维度标签）==========
    std::vector<int32_t> modeA;
    std::vector<int32_t> modeB;  // 仅 Trinary 使用
    std::vector<int32_t> modeC;
    std::vector<int32_t> modeD;

    // ========== 操作符 ==========
    // Elementwise Binary: D = opAC(α*opA(A), γ*opC(C))
    acltensorOperator_t opA  = ACLTENSOR_OP_IDENTITY;  // 阶段一只支持 IDENTITY
    acltensorOperator_t opC  = ACLTENSOR_OP_IDENTITY;  // 阶段一只支持 IDENTITY
    acltensorOperator_t opAC = ACLTENSOR_OP_ADD;       // 阶段一支持 ADD/MUL/MAX/MIN

    // Elementwise Trinary 专用
    acltensorOperator_t opB   = ACLTENSOR_OP_IDENTITY;
    acltensorOperator_t opAB  = ACLTENSOR_OP_ADD;
    acltensorOperator_t opABC = ACLTENSOR_OP_ADD;

    // ========== 计算精度 ==========
    acltensorComputeDescriptor_t descCompute;

    // ========== 用户属性 - TODO: Phase 2 ==========
    // uint32_t tag = 0;
    // float flops = 0.0f;
    // float movedBytes = 0.0f;
};
```

### 5.4 Plan 结构体

```cpp
// api/core/plan.hpp

#include <memory>

// 内核解决方案基类（抽象接口）
class KernelSolution {
public:
    virtual ~KernelSolution() = default;
    virtual uint64_t uid() const = 0;
    virtual size_t   workspaceSize() const { return 0; }  // Elementwise 通常不需要
    virtual bool     isValid() const = 0;
    virtual acltensorStatus_t execute(...) = 0;
};

// Elementwise Binary 专用解决方案
class ElementwiseBinarySolution : public KernelSolution {
public:
    using ExecuteFunc = acltensorStatus_t (*)(
        const void* A, const void* C, void* D,
        int64_t elemNum,
        const void* alpha, const void* gamma,
        aclrtStream stream);

    ElementwiseBinarySolution(uint64_t uid,
                               acltensorOperator_t op,
                               acltensorDataType_t type,
                               ExecuteFunc func);

    uint64_t uid() const override { return mUid; }
    bool isValid() const override { return mExecuteFunc != nullptr; }

    acltensorStatus_t execute(const void* alpha, const void* A,
                              const void* gamma, const void* C,
                              void* D, aclrtStream stream) override;

private:
    uint64_t              mUid;
    acltensorOperator_t   mOp;
    acltensorDataType_t   mDataType;
    ExecuteFunc           mExecuteFunc;
};

struct acltensorPlan {
    acltensorOperationDescriptor_t opDesc;
    acltensorPlanPreference_t      pref;
    uint64_t                       requiredWorkspace = 0;
    std::unique_ptr<KernelSolution> solution;
};
```

---

## 六、算子 Kernel 实现模板

### 6.1 Tiling 数据结构

```cpp
// src/add/arch35/add_tiling.h

#pragma once
#include <cstdint>

namespace acltensor {
namespace add {

struct AddTilingData {
    int64_t totalElements;     // 总元素数
    int64_t tileLength;        // 每个核处理的元素数
    int64_t totalLengthAligned; // 对齐后的总长度
    uint32_t usedCoreNum;      // 使用的核数
};

} // namespace add
} // namespace acltensor
```

### 6.2 Host 端实现

```cpp
// src/add/add_host.cpp

#include "add_host.h"
#include "arch35/add_tiling.h"
#include "acl/acl.h"

namespace acltensor {
namespace add {

// 计算 Tiling 数据
static void ComputeTiling(int64_t totalElements, uint32_t coreNum, AddTilingData& tiling) {
    tiling.totalElements = totalElements;
    tiling.usedCoreNum = std::min(coreNum, (uint32_t)((totalElements + 255) / 256));
    tiling.tileLength = (totalElements + tiling.usedCoreNum - 1) / tiling.usedCoreNum;
    tiling.totalLengthAligned = tiling.tileLength * tiling.usedCoreNum;
}

// Host 端入口函数
acltensorStatus_t AddF32(
    const void* A, const void* C, void* D,
    int64_t elemNum,
    const void* alpha, const void* gamma,
    aclrtStream stream)
{
    // 1. 计算 Tiling
    AddTilingData tiling;
    uint32_t coreNum = 32;  // 从设备信息获取
    ComputeTiling(elemNum, coreNum, tiling);

    // 2. 分配 Tiling 设备内存
    void* tilingDev = nullptr;
    aclrtMalloc(&tilingDev, sizeof(tiling), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(tilingDev, &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);

    // 3. 启动 Kernel
    uint32_t blockDim = tiling.usedCoreNum;
    // aclrtKernelLaunch(..., blockDim, stream);

    // 4. 清理
    aclrtFree(tilingDev);

    return ACLTENSOR_STATUS_SUCCESS;
}

// FP16 版本
acltensorStatus_t AddF16(
    const void* A, const void* C, void* D,
    int64_t elemNum,
    const void* alpha, const void* gamma,
    aclrtStream stream)
{
    // 类似实现...
    return ACLTENSOR_STATUS_SUCCESS;
}

} // namespace add
} // namespace acltensor
```

### 6.3 AscendC Kernel 实现

```cpp
// src/add/add_kernel.cpp

#include "kernel_operator.h"
#include "arch35/add_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BLOCK_LENGTH = 256;  // 每次处理的元素数

template <typename T>
class AddKernel {
public:
    __aicore__ inline AddKernel() {}

    __aicore__ inline void Init(GM_ADDR A, GM_ADDR C, GM_ADDR D,
                                  GM_ADDR tilingGm, TPipe* pipe) {
        // 1. 获取 Tiling 数据
        GM_ADDR tilingPtr = tilingGm;
        auto tilingData = (__gm__ acltensor::add::AddTilingData*)tilingPtr;
        totalElements = tilingData->totalElements;
        tileLength = tilingData->tileLength;

        // 2. 计算当前核处理的偏移
        blockIdx = GetBlockIdx();
        offset = blockIdx * tileLength;
        length = std::min(tileLength, totalElements - offset);

        // 3. 初始化 Global Tensor
        gmA.SetGlobalBuffer((__gm__ T*)A + offset, length);
        gmC.SetGlobalBuffer((__gm__ T*)C + offset, length);
        gmD.SetGlobalBuffer((__gm__ T*)D + offset, length);

        // 4. 初始化队列
        pipe->InitBuffer(inQueueA, BUFFER_NUM, BLOCK_LENGTH * sizeof(T));
        pipe->InitBuffer(inQueueC, BUFFER_NUM, BLOCK_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueD, BUFFER_NUM, BLOCK_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process() {
        int64_t loopCount = (length + BLOCK_LENGTH - 1) / BLOCK_LENGTH;

        for (int64_t i = 0; i < loopCount; ++i) {
            int64_t currentLength = std::min(BLOCK_LENGTH, length - i * BLOCK_LENGTH);

            // 拷入
            CopyIn(i * BLOCK_LENGTH, currentLength);
            // 计算
            Compute(currentLength);
            // 拷出
            CopyOut(i * BLOCK_LENGTH, currentLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t length) {
        LocalTensor<T> aLocal = inQueueA.AllocTensor<T>();
        LocalTensor<T> cLocal = inQueueC.AllocTensor<T>();

        DataCopy(aLocal, gmA[offset], length);
        DataCopy(cLocal, gmC[offset], length);

        inQueueA.EnQue(aLocal);
        inQueueC.EnQue(cLocal);
    }

    __aicore__ inline void Compute(int64_t length) {
        LocalTensor<T> aLocal = inQueueA.DeQue<T>();
        LocalTensor<T> cLocal = inQueueC.DeQue<T>();
        LocalTensor<T> dLocal = outQueueD.AllocTensor<T>();

        // D = A + C (忽略 alpha/gamma 简化)
        Add(dLocal, aLocal, cLocal, length);

        outQueueD.EnQue<T>(dLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueC.FreeTensor(cLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t length) {
        LocalTensor<T> dLocal = outQueueD.DeQue<T>();
        DataCopy(gmD[offset], dLocal, length);
        outQueueD.FreeTensor(dLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA, inQueueC;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueD;

    GlobalTensor<T> gmA, gmC, gmD;

    int64_t totalElements;
    int64_t tileLength;
    int64_t blockIdx;
    int64_t offset;
    int64_t length;
};

// Kernel 入口
extern "C" __global__ __aicore__ void add_kernel_f32(
    GM_ADDR A, GM_ADDR C, GM_ADDR D, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    AddKernel<float> op;
    op.Init(A, C, D, tilingGm, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void add_kernel_f16(
    GM_ADDR A, GM_ADDR C, GM_ADDR D, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    AddKernel<half> op;
    op.Init(A, C, D, tilingGm, &pipe);
    op.Process();
}
```

---

## 七、解决方案注册表

### 7.1 注册表设计

```cpp
// api/elementwise/elementwise_solution_registry.hpp

#pragma once

#include <vector>
#include <mutex>
#include "elementwise_solution.hpp"

namespace acltensor {

// 单例注册表
class ElementwiseSolutionRegistry {
public:
    static ElementwiseSolutionRegistry& instance() {
        static ElementwiseSolutionRegistry registry;
        return registry;
    }

    // 注册 Binary 解决方案
    void registerBinarySolution(std::unique_ptr<ElementwiseBinarySolution> solution) {
        std::lock_guard<std::mutex> lock(mMutex);
        mBinarySolutions.push_back(std::move(solution));
    }

    // 查找 Binary 解决方案
    ElementwiseBinarySolution* findBinarySolution(
        acltensorOperator_t op,
        acltensorDataType_t dataType)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        for (auto& sol : mBinarySolutions) {
            if (sol->match(op, dataType)) {
                return sol.get();
            }
        }
        return nullptr;
    }

    // 注册 Trinary 解决方案
    void registerTrinarySolution(std::unique_ptr<ElementwiseTrinarySolution> solution) {
        std::lock_guard<std::mutex> lock(mMutex);
        mTrinarySolutions.push_back(std::move(solution));
    }

    // 查找 Trinary 解决方案
    ElementwiseTrinarySolution* findTrinarySolution(
        acltensorOperator_t opAB,
        acltensorOperator_t opABC,
        acltensorDataType_t dataType);

private:
    ElementwiseSolutionRegistry() = default;

    std::mutex mMutex;
    std::vector<std::unique_ptr<ElementwiseBinarySolution>> mBinarySolutions;
    std::vector<std::unique_ptr<ElementwiseTrinarySolution>> mTrinarySolutions;
};

// 自动注册辅助宏
#define REGISTER_ELEMENTWISE_BINARY_SOLUTION(ClassName, Op, DataType, ExecuteFunc) \
    namespace { \
        struct ClassName##Registrar { \
            ClassName##Registrar() { \
                auto solution = std::make_unique<acltensor::ElementwiseBinarySolution>( \
                    ClassName##_uid, Op, DataType, ExecuteFunc); \
                acltensor::ElementwiseSolutionRegistry::instance() \
                    .registerBinarySolution(std::move(solution)); \
            } \
        }; \
        static ClassName##Registrar ClassName##_registrar; \
    }

} // namespace acltensor
```

### 7.2 解决方案注册实例

```cpp
// src/add/add_solution_registration.cpp

#include "api/elementwise/elementwise_solution_registry.hpp"
#include "add_host.h"

namespace acltensor {

// 生成唯一 ID
constexpr uint64_t ADD_F32_UID = 0x0001000100010001ULL;
constexpr uint64_t ADD_F16_UID = 0x0001000100010002ULL;

// 注册 Add FP32 解决方案
REGISTER_ELEMENTWISE_BINARY_SOLUTION(
    AddF32,
    ACLTENSOR_OP_ADD,
    ACLTENSOR_R_32F,
    add::AddF32
);

// 注册 Add FP16 解决方案
REGISTER_ELEMENTWISE_BINARY_SOLUTION(
    AddF16,
    ACLTENSOR_OP_ADD,
    ACLTENSOR_R_16F,
    add::AddF16
);

} // namespace acltensor
```

---

## 八、验证方案

### 8.1 测试用例

```cpp
// tests/test_elementwise_binary.cpp

#include "cann_ops_tensor.h"
#include "test_common.h"
#include <iostream>
#include <vector>
#include <cmath>

// 测试 Elementwise Binary Add
bool test_elementwise_binary_add() {
    const int64_t M = 1024, N = 1024;
    const int64_t totalElements = M * N;

    // 1. 创建句柄
    acltensorHandle_t handle;
    CHECK_STATUS(acltensorCreate(&handle));

    // 2. 创建张量描述符
    int64_t extent[] = {M, N};
    int32_t modeA[] = {0, 1};
    int32_t modeC[] = {0, 1};
    int32_t modeD[] = {0, 1};

    acltensorTensorDescriptor_t descA, descC, descD;
    CHECK_STATUS(acltensorCreateTensorDescriptor(handle, &descA, 2, extent, NULL, ACLTENSOR_R_32F, 4));
    CHECK_STATUS(acltensorCreateTensorDescriptor(handle, &descC, 2, extent, NULL, ACLTENSOR_R_32F, 4));
    CHECK_STATUS(acltensorCreateTensorDescriptor(handle, &descD, 2, extent, NULL, ACLTENSOR_R_32F, 4));

    // 3. 创建 Elementwise Binary 操作描述符
    acltensorOperationDescriptor_t opDesc;
    CHECK_STATUS(acltensorCreateElementwiseBinary(handle, &opDesc,
        descA, modeA, ACLTENSOR_OP_IDENTITY,
        descC, modeC, ACLTENSOR_OP_IDENTITY,
        descD, modeD, ACLTENSOR_OP_ADD,
        ACLTENSOR_COMPUTE_DESC_32F));

    // 4. 创建 Plan
    acltensorPlanPreference_t pref;
    CHECK_STATUS(acltensorCreatePlanPreference(handle, &pref, ACLTENSOR_ALGO_DEFAULT));

    acltensorPlan_t plan;
    CHECK_STATUS(acltensorCreatePlan(handle, &plan, opDesc, pref, 0));

    // 5. 准备数据
    std::vector<float> h_A(totalElements);
    std::vector<float> h_C(totalElements);
    std::vector<float> h_D(totalElements);

    for (int64_t i = 0; i < totalElements; ++i) {
        h_A[i] = static_cast<float>(i);
        h_C[i] = static_cast<float>(i * 2);
    }

    // 6. 分配设备内存
    void *d_A, *d_C, *d_D;
    aclrtMalloc(&d_A, totalElements * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_C, totalElements * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_D, totalElements * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(d_A, h_A.data(), totalElements * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_C, h_C.data(), totalElements * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    // 7. 执行
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float alpha = 1.0f, gamma = 1.0f;
    CHECK_STATUS(acltensorElementwiseBinaryExecute(handle, plan,
        &alpha, d_A, &gamma, d_C, d_D, stream));

    aclrtSynchronizeStream(stream);

    // 8. 验证结果
    aclrtMemcpy(h_D.data(), d_D, totalElements * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    bool passed = true;
    for (int64_t i = 0; i < totalElements; ++i) {
        float expected = h_A[i] + h_C[i];
        if (std::abs(h_D[i] - expected) > 1e-5) {
            passed = false;
            std::cout << "Mismatch at " << i << ": got " << h_D[i]
                      << ", expected " << expected << std::endl;
            break;
        }
    }

    // 9. 清理
    aclrtFree(d_A);
    aclrtFree(d_C);
    aclrtFree(d_D);
    aclrtDestroyStream(stream);

    acltensorDestroyPlan(plan);
    acltensorDestroyPlanPreference(pref);
    acltensorDestroyOperationDescriptor(opDesc);
    acltensorDestroyTensorDescriptor(descA);
    acltensorDestroyTensorDescriptor(descC);
    acltensorDestroyTensorDescriptor(descD);
    acltensorDestroy(handle);

    return passed;
}

int main() {
    aclInit(nullptr);

    std::cout << "test_elementwise_binary_add: "
              << (test_elementwise_binary_add() ? "PASSED" : "FAILED")
              << std::endl;

    aclFinalize();
    return 0;
}
```

---

## 九、任务分解与优先级

### 9.1 核心任务列表

| 任务ID | 任务描述 | 依赖 | 优先级 | 预估代码量 |
|--------|---------|------|--------|-----------|
| T1 | 创建头文件框架 `cann_ops_tensor_types.h` / `cann_ops_tensor.h` | 无 | P0 | 0.2K |
| T2 | 实现 Handle 管理 (`acltensorCreate`/`Destroy`) | T1 | P0 | 0.2K |
| T3 | 实现 TensorDescriptor 管理 (仅 FP32) | T1 | P0 | 0.2K |
| T4 | 实现 OperationDescriptor (仅 Elementwise Binary) | T3 | P0 | 0.3K |
| T5 | 实现 PlanPreference (简化版) | T1 | P0 | 0.1K |
| T6 | 实现 Plan 创建和内核选择逻辑 | T4, T5 | P0 | 0.2K |
| T7 | 实现 Elementwise Binary 框架层 | T6 | P0 | 0.3K |
| T8 | 实现 ADD 算子 (仅 FP32) | T7 | P0 | 0.3K |
| T9 | 编写单元测试 | T8 | P0 | 0.3K |
| T10 | CMake 构建脚本 | T1 | P0 | 0.1K |

### 9.2 里程碑

```
Milestone 1 (T1-T6): 核心框架搭建
    ├── 头文件定义（精简版，仅 FP32）
    ├── Handle 管理
    ├── TensorDescriptor（仅 FP32）
    ├── OperationDescriptor（仅 Binary）
    └── Plan 管理

Milestone 2 (T7): Elementwise Binary 框架层
    └── Binary 分发逻辑

Milestone 3 (T8): ADD 算子实现
    └── ADD Kernel (FP32)

Milestone 4 (T9-T10): 测试与构建
    ├── 单元测试（端到端验证）
    └── CMake 脚本
```

### 9.3 后续阶段任务规划

| 阶段 | 任务ID | 任务描述 |
|------|--------|---------|
| **Phase 2** | T11 | 扩展数据类型支持 (FP16) |
| | T12 | 实现 MUL 算子 |
| | T13 | 实现 SUB 算子 |
| | T14 | 实现 DIV 算子 |
| | T15 | 实现 MAX/MIN 算子 |
| | T16 | 实现 Elementwise Trinary 框架层 |
| **Phase 3** | T17 | 一元操作符支持 (SQRT/RELU/EXP/LOG 等) |
| | T18 | PlanCache 机制 |
| | T19 | 日志系统 |
| **Phase 4** | T20 | Contraction 操作 |
| | T21 | MatMul 算子 |

---

## 十、后续阶段规划

### 10.1 阶段二：Elementwise 扩展

**目标**：扩展 Binary 算子和数据类型

| 内容 | 说明 |
|------|------|
| 扩展算子 | MUL, SUB, DIV, MAX, MIN |
| 扩展数据类型 | FP16 支持 |
| Trinary 框架 | `acltensorCreateElementwiseTrinary`, `acltensorElementwiseTrinaryExecute` |
| PlanCache | 完整缓存机制 |
| 日志系统 | 全部 6 个接口 |

### 10.2 阶段三：Contraction

**目标**：实现张量收缩操作

| 内容 | 说明 |
|------|------|
| 接口 | `acltensorCreateContraction`, `acltensorContract` |
| 算子 | MatMul (FP32/FP16) |
| 一元操作符 | SQRT, RELU, EXP, LOG, ABS, NEG 等 |

### 10.3 阶段四：Reduction

**目标**：实现张量归约操作

| 内容 | 说明 |
|------|------|
| 接口 | `acltensorCreateReduction`, `acltensorReduce` |
| 算子 | ReduceSum, ReduceMax, ReduceMin |
| 激活函数 | GELU, SILU |

### 10.4 阶段五：Permutation + 扩展

**目标**：实现张量置换操作及全面扩展

| 内容 | 说明 |
|------|------|
| 接口 | `acltensorCreatePermutation`, `acltensorPermute` |
| 算子 | Transpose |
| 扩展数据类型 | BF16, INT8, INT32, FP64 |
| Complex 支持 | Complex FP32, Complex FP64 |

---

## 附录：参考资源

1. **hipTensor 源码**: `C:\Users\zl\Desktop\gitee\tmp\rocm-libraries-develop\projects\hiptensor`
2. **设计方案**: `C:\Users\zl\Desktop\gitee\cann_opensource\master\ops-fft\docs\zh\design\ops_tensor_api_design.md`
3. **AscendC 文档**: 华为昇腾 AscendC 编程指南
4. **CANN ACL 文档**: Ascend Computing Language API 参考
