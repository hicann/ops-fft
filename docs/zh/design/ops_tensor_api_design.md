# CANN ops-tensor 接口设计方案

本文档基于 hipTensor 的 API 设计，为 CANN ops-tensor 库设计一套完整的张量运算接口。

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CANN ops-tensor API 层次结构                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     用户应用 (User Application)                   │   │
│  └────────────────────────────┬────────────────────────────────────┘   │
│                               │                                         │
│  ┌────────────────────────────▼────────────────────────────────────┐   │
│  │                 高级 API (直接执行)                               │   │
│  │   acltensorContraction(), acltensorReduce(), acltensorPermute()  │   │
│  └────────────────────────────┬────────────────────────────────────┘   │
│                               │                                         │
│  ┌────────────────────────────▼────────────────────────────────────┐   │
│  │                 核心对象 (描述符 + Plan)                          │   │
│  │   Handle → TensorDesc → OperationDesc → PlanPreference → Plan    │   │
│  └────────────────────────────┬────────────────────────────────────┘   │
│                               │                                         │
│  ┌────────────────────────────▼────────────────────────────────────┐   │
│  │                 底层内核 (Kernel Library)                         │   │
│  │   AscendC Kernel Library (基于 Ascend AI Core)                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、核心数据类型定义

### 2.1 头文件：`include/cann_ops_tensor_types.h`

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
    ACLTENSOR_STATUS_SUCCESS = 0,               /* 操作成功 */
    ACLTENSOR_STATUS_NOT_INITIALIZED = 1,       /* 句柄未初始化 */
    ACLTENSOR_STATUS_ALLOC_FAILED = 3,          /* 内存分配失败 */
    ACLTENSOR_STATUS_INVALID_VALUE = 7,         /* 无效参数值 */
    ACLTENSOR_STATUS_ARCH_MISMATCH = 8,         /* 架构不支持 */
    ACLTENSOR_STATUS_EXECUTION_FAILED = 13,     /* 内核执行失败 */
    ACLTENSOR_STATUS_INTERNAL_ERROR = 14,       /* 内部错误 */
    ACLTENSOR_STATUS_NOT_SUPPORTED = 15,        /* 不支持的操作 */
    ACLTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19, /* 工作空间不足 */
    ACLTENSOR_STATUS_INSUFFICIENT_DRIVER = 20,  /* 驱动版本不足 */
    ACLTENSOR_STATUS_IO_ERROR = 21,             /* 文件 I/O 错误 */
} acltensorStatus_t;

/*============================================================================
 * 2. 数据类型枚举
 *============================================================================*/
typedef enum {
    ACLTENSOR_R_16F = 0,     /* FP16 半精度浮点 */
    ACLTENSOR_R_32F = 1,     /* FP32 单精度浮点 */
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
    ACLTENSOR_COMPUTE_DESC_16F = (1U << 0),   /* FP16 计算 */
    ACLTENSOR_COMPUTE_DESC_32F = (1U << 2),   /* FP32 计算 */
    ACLTENSOR_COMPUTE_DESC_64F = (1U << 4),   /* FP64 计算 */
    ACLTENSOR_COMPUTE_DESC_16BF = (1U << 10), /* BF16 计算 */
    ACLTENSOR_COMPUTE_DESC_C32F = (1U << 11), /* Complex FP32 计算 */
    ACLTENSOR_COMPUTE_DESC_C64F = (1U << 12), /* Complex FP64 计算 */
    ACLTENSOR_COMPUTE_DESC_NONE = 0,          /* 无指定 */
} acltensorComputeDescriptor_t;

/*============================================================================
 * 4. 操作符枚举
 *============================================================================*/
/* 一元操作符 */
typedef enum {
    ACLTENSOR_OP_IDENTITY = 1,  /* 恒等: y = x */
    ACLTENSOR_OP_SQRT = 2,      /* 平方根: y = sqrt(x) */
    ACLTENSOR_OP_RELU = 8,      /* ReLU: y = max(0, x) */
    ACLTENSOR_OP_CONJ = 9,      /* 共轭: y = conj(x) */
    ACLTENSOR_OP_RCP = 10,      /* 倒数: y = 1/x */
    ACLTENSOR_OP_SIGMOID = 11,  /* Sigmoid: y = 1/(1+e^(-x)) */
    ACLTENSOR_OP_TANH = 12,     /* Tanh: y = tanh(x) */
    ACLTENSOR_OP_EXP = 22,      /* 指数: y = e^x */
    ACLTENSOR_OP_LOG = 23,      /* 对数: y = ln(x) */
    ACLTENSOR_OP_ABS = 24,      /* 绝对值: y = |x| */
    ACLTENSOR_OP_NEG = 25,      /* 取负: y = -x */
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

    /* 二元操作符 */
    ACLTENSOR_OP_ADD = 3,       /* 加法: y = a + b */
    ACLTENSOR_OP_MUL = 5,       /* 乘法: y = a * b */
    ACLTENSOR_OP_MAX = 6,       /* 最大值: y = max(a, b) */
    ACLTENSOR_OP_MIN = 7,       /* 最小值: y = min(a, b) */
    ACLTENSOR_OP_SUB = 50,      /* 减法: y = a - b */
    ACLTENSOR_OP_DIV = 51,      /* 除法: y = a / b */
    ACLTENSOR_OP_POW = 52,      /* 幂运算: y = a^b */

    ACLTENSOR_OP_UNKNOWN = 126,
} acltensorOperator_t;

/*============================================================================
 * 5. 算法选择
 *============================================================================*/
typedef enum {
    ACLTENSOR_ALGO_DEFAULT = -1,          /* 启发式自动选择 */
    ACLTENSOR_ALGO_DEFAULT_PATIENT = -6,  /* 更精确但耗时的选择 */
} acltensorAlgo_t;

/*============================================================================
 * 6. 工作空间偏好
 *============================================================================*/
typedef enum {
    ACLTENSOR_WORKSPACE_MIN = 1,     /* 最小工作空间，至少一个算法可用 */
    ACLTENSOR_WORKSPACE_DEFAULT = 2, /* 默认工作空间 */
    ACLTENSOR_WORKSPACE_MAX = 3,     /* 最大工作空间，所有算法可用 */
} acltensorWorksizePreference_t;

/*============================================================================
 * 7. 缓存模式
 *============================================================================*/
typedef enum {
    ACLTENSOR_CACHE_MODE_NONE = 0,     /* 不使用缓存 */
    ACLTENSOR_CACHE_MODE_PEDANTIC = 1, /* 使用缓存（推荐） */
} acltensorCacheMode_t;

/*============================================================================
 * 8. 日志级别
 *============================================================================*/
typedef enum {
    ACLTENSOR_LOG_LEVEL_OFF = 0,              /* 关闭日志 */
    ACLTENSOR_LOG_LEVEL_ERROR = 1,            /* 错误日志 */
    ACLTENSOR_LOG_LEVEL_PERF_TRACE = 2,       /* 性能跟踪 */
    ACLTENSOR_LOG_LEVEL_PERF_HINT = 4,        /* 性能提示 */
    ACLTENSOR_LOG_LEVEL_HEURISTICS_TRACE = 8, /* 启发式选择日志 */
    ACLTENSOR_LOG_LEVEL_API_TRACE = 16,       /* API 调用跟踪 */
} acltensorLogLevel_t;

/*============================================================================
 * 9. 操作描述符属性
 *============================================================================*/
typedef enum {
    ACLTENSOR_OPERATION_DESCRIPTOR_TAG = 0,           /* 用户标签 */
    ACLTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE = 1,   /* 标量类型 */
    ACLTENSOR_OPERATION_DESCRIPTOR_FLOPS = 2,         /* FLOPS 估算 */
    ACLTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES = 3,   /* 数据移动字节数 */
    ACLTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT = 4,  /* 左填充 */
    ACLTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT = 5, /* 右填充 */
} acltensorOperationDescriptorAttribute_t;

/*============================================================================
 * 10. Plan 属性
 *============================================================================*/
typedef enum {
    ACLTENSOR_PLAN_REQUIRED_WORKSPACE = 0, /* 所需工作空间大小 */
    ACLTENSOR_PLAN_KERNEL_ID = 1,           /* 选中的内核 ID */
} acltensorPlanAttribute_t;

/*============================================================================
 * 11. PlanPreference 属性
 *============================================================================*/
typedef enum {
    ACLTENSOR_PLAN_PREFERENCE_CACHE_MODE = 0,     /* 缓存模式 */
    ACLTENSOR_PLAN_PREFERENCE_ALGO = 1,           /* 算法选择 */
    ACLTENSOR_PLAN_PREFERENCE_KERNEL_RANK = 2,    /* 内核秩 */
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
 * 13. 日志回调函数类型
 *============================================================================*/
typedef void (*acltensorLoggerCallback_t)(int32_t logContext,
                                          const char* funcName,
                                          const char* msg);

#ifdef __cplusplus
}
#endif

#endif /* CANN_OPS_TENSOR_TYPES_H */
```

---

## 三、主 API 头文件

### 3.1 头文件：`include/cann_ops_tensor.h`

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

/**
 * @brief 调整 Plan 缓存大小
 * @param[in] handle     库句柄
 * @param[in] numEntries 缓存条目数
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
acltensorStatus_t acltensorHandleResizePlanCache(acltensorHandle_t handle,
                                                  const uint32_t numEntries);

/**
 * @brief 将 Plan 缓存写入文件
 * @param[in] handle   库句柄
 * @param[in] fileName 文件名
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
acltensorStatus_t acltensorHandleWritePlanCacheToFile(const acltensorHandle_t handle,
                                                       const char fileName[]);

/**
 * @brief 从文件读取 Plan 缓存
 * @param[in]  handle           库句柄
 * @param[in]  fileName         文件名
 * @param[out] numCachelinesRead 读取的条目数
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
acltensorStatus_t acltensorHandleReadPlanCacheFromFile(acltensorHandle_t handle,
                                                        const char fileName[],
                                                        uint32_t* numCachelinesRead);

/*============================================================================
 *                        2. 张量描述符管理 (Tensor Descriptor)
 *============================================================================*/

/**
 * @brief 创建张量描述符
 * @param[in]  handle              库句柄
 * @param[out] desc                返回的张量描述符
 * @param[in]  numModes            维度数量
 * @param[in]  lens                各维度长度数组
 * @param[in]  strides             各维度步长数组（NULL 使用默认列主序）
 * @param[in]  dataType            数据类型
 * @param[in]  alignmentRequirement 内存对齐要求（字节）
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
acltensorStatus_t acltensorCreateTensorDescriptor(
    const acltensorHandle_t      handle,
    acltensorTensorDescriptor_t* desc,
    const uint32_t               numModes,
    const int64_t                lens[],
    const int64_t                strides[],
    acltensorDataType_t          dataType,
    uint32_t                     alignmentRequirement);

/**
 * @brief 销毁张量描述符
 * @param[in] desc 要销毁的张量描述符
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
acltensorStatus_t acltensorDestroyTensorDescriptor(acltensorTensorDescriptor_t desc);

/*============================================================================
 *                        3. 操作描述符管理 (Operation Descriptor)
 *============================================================================*/

/**
 * @brief 创建张量收缩操作描述符: D = α*A*B + β*C
 * @param[in]  handle      库句柄
 * @param[out] desc        返回的操作描述符
 * @param[in]  descA       张量 A 描述符
 * @param[in]  modeA       张量 A 的模式数组
 * @param[in]  opA         张量 A 的预处理操作符
 * @param[in]  descB       张量 B 描述符
 * @param[in]  modeB       张量 B 的模式数组
 * @param[in]  opB         张量 B 的预处理操作符
 * @param[in]  descC       张量 C 描述符
 * @param[in]  modeC       张量 C 的模式数组
 * @param[in]  opC         张量 C 的预处理操作符
 * @param[in]  descD       张量 D 描述符（输出）
 * @param[in]  modeD       张量 D 的模式数组
 * @param[in]  descCompute 计算精度
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
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

/**
 * @brief 创建张量归约操作描述符: D = α*opReduce(opA(A)) + β*opC(C)
 * @param[in]  handle      库句柄
 * @param[out] desc        返回的操作描述符
 * @param[in]  descA       张量 A 描述符
 * @param[in]  modeA       张量 A 的模式数组
 * @param[in]  opA         张量 A 的预处理操作符
 * @param[in]  descC       张量 C 描述符
 * @param[in]  modeC       张量 C 的模式数组
 * @param[in]  opC         张量 C 的预处理操作符
 * @param[in]  descD       张量 D 描述符（输出）
 * @param[in]  modeD       张量 D 的模式数组
 * @param[in]  opReduce    归约操作符（如 ADD/MAX/MIN）
 * @param[in]  descCompute 计算精度
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
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

/**
 * @brief 创建张量置换操作描述符: B = α*opA(A)
 * @param[in]  handle      库句柄
 * @param[out] desc        返回的操作描述符
 * @param[in]  descA       输入张量 A 描述符
 * @param[in]  modeA       张量 A 的模式数组
 * @param[in]  opA         预处理操作符
 * @param[in]  descB       输出张量 B 描述符
 * @param[in]  modeB       张量 B 的模式数组
 * @param[in]  descCompute 计算精度
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
acltensorStatus_t acltensorCreatePermutation(
    const acltensorHandle_t            handle,
    acltensorOperationDescriptor_t*    desc,
    const acltensorTensorDescriptor_t  descA,
    const int32_t                      modeA[],
    acltensorOperator_t                opA,
    const acltensorTensorDescriptor_t  descB,
    const int32_t                      modeB[],
    const acltensorComputeDescriptor_t descCompute);

/**
 * @brief 创建二元逐元素操作描述符: D = opAC(α*opA(A), γ*opC(C))
 * @param[in]  handle      库句柄
 * @param[out] desc        返回的操作描述符
 * @param[in]  descA       张量 A 描述符
 * @param[in]  modeA       张量 A 的模式数组
 * @param[in]  opA         张量 A 的预处理操作符
 * @param[in]  descC       张量 C 描述符
 * @param[in]  modeC       张量 C 的模式数组
 * @param[in]  opC         张量 C 的预处理操作符
 * @param[in]  descD       输出张量 D 描述符
 * @param[in]  modeD       张量 D 的模式数组
 * @param[in]  opAC        二元操作符
 * @param[in]  descCompute 计算精度
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
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

/**
 * @brief 创建三元逐元素操作描述符: D = opABC(opAB(α*A,β*B), γ*C)
 * @param[in]  handle      库句柄
 * @param[out] desc        返回的操作描述符
 * @param[in]  descA       张量 A 描述符
 * @param[in]  modeA       张量 A 的模式数组
 * @param[in]  opA         张量 A 的预处理操作符
 * @param[in]  descB       张量 B 描述符
 * @param[in]  modeB       张量 B 的模式数组
 * @param[in]  opB         张量 B 的预处理操作符
 * @param[in]  descC       张量 C 描述符
 * @param[in]  modeC       张量 C 的模式数组
 * @param[in]  opC         张量 C 的预处理操作符
 * @param[in]  descD       输出张量 D 描述符
 * @param[in]  modeD       张量 D 的模式数组
 * @param[in]  opAB        第一个二元操作符
 * @param[in]  opABC       第二个二元操作符
 * @param[in]  descCompute 计算精度
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
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

/**
 * @brief 销毁操作描述符
 * @param[in] desc 要销毁的操作描述符
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
acltensorStatus_t acltensorDestroyOperationDescriptor(acltensorOperationDescriptor_t desc);

/**
 * @brief 设置操作描述符属性
 */
acltensorStatus_t acltensorOperationDescriptorSetAttribute(
    const acltensorHandle_t                 handle,
    acltensorOperationDescriptor_t          desc,
    acltensorOperationDescriptorAttribute_t attr,
    const void*                             buf,
    size_t                                  sizeInBytes);

/**
 * @brief 获取操作描述符属性
 */
acltensorStatus_t acltensorOperationDescriptorGetAttribute(
    const acltensorHandle_t                 handle,
    acltensorOperationDescriptor_t          desc,
    acltensorOperationDescriptorAttribute_t attr,
    void*                                   buf,
    size_t                                  sizeInBytes);

/*============================================================================
 *                        4. Plan 和 PlanPreference 管理
 *============================================================================*/

/**
 * @brief 创建 Plan 偏好设置
 * @param[in]  handle  库句柄
 * @param[out] pref    返回的 PlanPreference
 * @param[in]  algo    算法选择
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
acltensorStatus_t acltensorCreatePlanPreference(
    const acltensorHandle_t    handle,
    acltensorPlanPreference_t* pref,
    acltensorAlgo_t            algo);

/**
 * @brief 销毁 Plan 偏好设置
 */
acltensorStatus_t acltensorDestroyPlanPreference(acltensorPlanPreference_t pref);

/**
 * @brief 设置 Plan 偏好属性
 */
acltensorStatus_t acltensorPlanPreferenceSetAttribute(
    const acltensorHandle_t            handle,
    acltensorPlanPreference_t          pref,
    acltensorPlanPreferenceAttribute_t attr,
    const void*                        buf,
    size_t                             sizeInBytes);

/**
 * @brief 估算工作空间大小
 * @param[in]  handle              库句柄
 * @param[in]  desc                操作描述符
 * @param[in]  planPref            Plan 偏好（可为 NULL）
 * @param[in]  workspacePref       工作空间偏好
 * @param[out] workspaceSizeEstimate 估算的工作空间大小
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
acltensorStatus_t acltensorEstimateWorkspaceSize(
    const acltensorHandle_t              handle,
    const acltensorOperationDescriptor_t desc,
    const acltensorPlanPreference_t      planPref,
    const acltensorWorksizePreference_t  workspacePref,
    uint64_t*                            workspaceSizeEstimate);

/**
 * @brief 创建执行计划
 * @param[in]  handle             库句柄
 * @param[out] plan               返回的 Plan
 * @param[in]  desc               操作描述符
 * @param[in]  pref               Plan 偏好（可为 NULL 使用默认）
 * @param[in]  workspaceSizeLimit 工作空间大小限制
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
acltensorStatus_t acltensorCreatePlan(
    const acltensorHandle_t              handle,
    acltensorPlan_t*                     plan,
    const acltensorOperationDescriptor_t desc,
    const acltensorPlanPreference_t      pref,
    uint64_t                             workspaceSizeLimit);

/**
 * @brief 销毁执行计划
 */
acltensorStatus_t acltensorDestroyPlan(acltensorPlan_t plan);

/**
 * @brief 获取 Plan 属性
 */
acltensorStatus_t acltensorPlanGetAttribute(
    const acltensorHandle_t  handle,
    const acltensorPlan_t    plan,
    acltensorPlanAttribute_t attr,
    void*                    buf,
    size_t                   sizeInBytes);

/*============================================================================
 *                        5. 执行函数 (Execution)
 *============================================================================*/

/**
 * @brief 执行张量收缩: D = α*A*B + β*C
 * @param[in]  handle        库句柄
 * @param[in]  plan          执行计划
 * @param[in]  alpha         A*B 的缩放因子（主机内存）
 * @param[in]  A             张量 A 数据（设备内存）
 * @param[in]  B             张量 B 数据（设备内存）
 * @param[in]  beta          C 的缩放因子（主机内存）
 * @param[in]  C             张量 C 数据（设备内存）
 * @param[out] D             输出张量 D（设备内存）
 * @param[in]  workspace     工作空间（可为 NULL）
 * @param[in]  workspaceSize 工作空间大小
 * @param[in]  stream        ACL 流
 * @return ACLTENSOR_STATUS_SUCCESS 成功
 */
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

/**
 * @brief 执行张量归约: D = α*opReduce(A) + β*C
 */
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

/**
 * @brief 执行张量置换: B = α*A
 */
acltensorStatus_t acltensorPermute(
    const acltensorHandle_t handle,
    const acltensorPlan_t   plan,
    const void*             alpha,
    const void*             A,
    void*                   B,
    aclrtStream             stream);

/**
 * @brief 执行二元逐元素操作: D = opAC(α*A, γ*C)
 */
acltensorStatus_t acltensorElementwiseBinaryExecute(
    const acltensorHandle_t handle,
    const acltensorPlan_t   plan,
    const void*             alpha,
    const void*             A,
    const void*             gamma,
    const void*             C,
    void*                   D,
    aclrtStream             stream);

/**
 * @brief 执行三元逐元素操作: D = opABC(opAB(α*A,β*B), γ*C)
 */
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

/*============================================================================
 *                        6. 辅助工具函数 (Utilities)
 *============================================================================*/

/**
 * @brief 获取错误码对应的描述字符串
 * @param[in] error 错误码
 * @return 描述字符串
 */
const char* acltensorGetErrorString(const acltensorStatus_t error);

/**
 * @brief 获取库版本号
 * @return 版本号 (major*10000 + minor*100 + patch)
 */
size_t acltensorGetVersion(void);

/*============================================================================
 *                        7. 日志系统 (Logging)
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

## 四、内部实现结构设计

### 4.1 目录结构

```
ops-tensor/
├── include/                          # 公共头文件（对外）
│   ├── cann_ops_tensor.h             # 主 API 入口
│   └── cann_ops_tensor_types.h       # 类型定义（枚举、结构体）
│
├── api/                              # 对外接口 + 上层框架实现
│   ├── core/                         # 核心框架
│   │   ├── handle.cpp                # 句柄管理
│   │   ├── handle.hpp
│   │   ├── handle_impl.hpp           # 句柄内部实现
│   │   ├── tensor_descriptor.cpp     # 张量描述符
│   │   ├── tensor_descriptor.hpp
│   │   ├── operation_descriptor.cpp  # 操作描述符
│   │   ├── operation_descriptor.hpp
│   │   ├── plan.cpp                  # Plan 管理
│   │   ├── plan.hpp
│   │   ├── plan_preference.cpp       # Plan 偏好设置
│   │   ├── plan_preference.hpp
│   │   ├── plan_cache.cpp            # Plan 缓存（LRU）
│   │   ├── plan_cache.hpp
│   │   └── device_info.hpp           # 设备信息查询
│   │
│   ├── elementwise/                  # Elementwise 操作框架（调度层）
│   │   ├── elementwise_binary.cpp    # Binary: D = opAC(α*A, γ*C)
│   │   ├── elementwise_binary.hpp
│   │   ├── elementwise_trinary.cpp   # Trinary: D = opABC(opAB(α*A,β*B), γ*C)
│   │   ├── elementwise_trinary.hpp
│   │   └── elementwise_impl.hpp      # 内部实现辅助
│   │
│   ├── contraction/                  # Contraction 操作框架
│   │   ├── contraction.cpp           # D = α*A*B + β*C
│   │   ├── contraction.hpp
│   │   ├── contraction_impl.hpp
│   │   └── contraction_solution.hpp  # 解决方案抽象
│   │
│   ├── reduction/                    # Reduction 操作框架
│   │   ├── reduction.cpp             # D = α*reduce(A) + β*C
│   │   ├── reduction.hpp
│   │   └── reduction_impl.hpp
│   │
│   ├── permutation/                  # Permutation 操作框架
│   │   ├── permutation.cpp           # B = α*A
│   │   ├── permutation.hpp
│   │   └── permutation_impl.hpp
│   │
│   └── utils/                        # 公共工具函数
│       ├── tiling_utils.cpp          # Tiling 计算工具
│       ├── tiling_utils.hpp
│       ├── memory_utils.cpp          # 设备内存管理工具
│       └── memory_utils.hpp
│
├── src/                              # 算子 Kernel 实现（按算子名组织）
│   ├── add/                          # Add 算子
│   │   ├── arch35/
│   │   │   └── add_struct.h          # Tiling 数据结构
│   │   ├── add_host.cpp              # 算子 Host 端（tiling计算 + kernel调度）
│   │   └── add_kernel.cpp            # AscendC Kernel 实现
│   │
│   ├── mul/                          # Mul 算子
│   │   ├── arch35/
│   │   │   └── mul_struct.h
│   │   ├── mul_host.cpp
│   │   └── mul_kernel.cpp
│   │
│   ├── sub/                          # Sub 算子
│   │   ├── arch35/
│   │   │   └── sub_struct.h
│   │   ├── sub_host.cpp
│   │   └── sub_kernel.cpp
│   │
│   ├── div/                          # Div 算子
│   │   ├── arch35/
│   │   │   └── div_struct.h
│   │   ├── div_host.cpp
│   │   └── div_kernel.cpp
│   │
│   ├── matmul/                       # MatMul 算子
│   │   ├── arch35/
│   │   │   └── matmul_struct.h
│   │   ├── matmul_host.cpp
│   │   └── matmul_kernel.cpp
│   │
│   ├── reduce_sum/                   # ReduceSum 算子
│   │   ├── arch35/
│   │   │   └── reduce_sum_struct.h
│   │   ├── reduce_sum_host.cpp
│   │   └── reduce_sum_kernel.cpp
│   │
│   ├── transpose/                    # Transpose 算子
│   │   ├── arch35/
│   │   │   └── transpose_struct.h
│   │   ├── transpose_host.cpp
│   │   └── transpose_kernel.cpp
│   │
│   └── max/                          # Max 算子（逐元素）
│       ├── arch35/
│       │   └── max_struct.h
│       ├── max_host.cpp
│       └── max_kernel.cpp
│
├── tests/                            # 测试代码
│   ├── test_common.cpp
│   ├── test_common.h
│   ├── test_handle.cpp
│   ├── test_elementwise.cpp
│   └── test_contraction.cpp
│
├── CMakeLists.txt
└── README.md
```

### 4.2 目录结构说明

#### 4.2.1 层次关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           用户调用                                       │
│                    acltensorElementwiseBinaryExecute()                  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        api/elementwise/                                 │
│                     Elementwise 操作框架层                               │
│                                                                         │
│   职责：                                                                │
│   - 解析操作描述符                                                      │
│   - 根据操作符类型分发到具体算子                                        │
│   - 管理通用 tiling 策略                                                │
│   - 处理 broadcast 语义                                                 │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        src/{算子名}/                                     │
│                       具体算子实现                                       │
│                                                                         │
│   {算子名}_host.cpp:                                                    │
│   - 算子特定的 tiling 计算                                              │
│   - 设备内存分配/拷贝                                                   │
│   - 调用 AscendC Kernel                                                 │
│                                                                         │
│   {算子名}_kernel.cpp:                                                  │
│   - AscendC Kernel 实现                                                 │
│   - 多核并行处理                                                        │
│   - UB 内存管理                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.2.2 职责划分

| 目录 | 职责 | 运行环境 |
|------|------|----------|
| `include/` | 公共类型定义，对外 API 声明 | 无 |
| `api/core/` | Handle、Descriptor、Plan 等核心对象管理 | Host (CPU) |
| `api/{操作类型}/` | 操作框架层，解析参数、分发算子 | Host (CPU) |
| `api/utils/` | 通用工具：内存管理、tiling 工具 | Host (CPU) |
| `src/{算子名}/` | 算子 Host 端 + AscendC Kernel | Host + NPU |

#### 4.2.3 算子目录结构模板

```
src/{算子名}/
├── arch35/                        # 架构相关（可扩展其他架构）
│   └── {算子名}_struct.h          # Tiling 数据结构定义
├── {算子名}_host.cpp               # Host 端实现
└── {算子名}_kernel.cpp             # AscendC Kernel 实现
```

**{算子名}_host.cpp 职责**：
- 参数验证
- Tiling 数据计算
- 设备内存分配/释放
- 数据 Host ↔ Device 拷贝
- 调用 Kernel 入口函数

**{算子名}_kernel.cpp 职责**：
- AscendC Kernel 类实现
- 多核数据切分
- UB 内存管理
- 向量计算

**arch35/{算子名}_struct.h 职责**：
- TilingData 结构体定义
- Kernel 需要的所有参数

### 4.3 核心结构体内部定义

```cpp
// api/core/handle_impl.hpp
#include <mutex>
#include <memory>

namespace acltensor {

// 设备信息类
class AscendDevice {
public:
    AscendDevice();
    ~AscendDevice() = default;

    int32_t getDeviceId() const { return mDeviceId; }
    uint32_t getCoreNum() const { return mCoreNum; }
    uint64_t getUbSize() const { return mUbSize; }
    bool supportsFp64() const { return mSupportsFp64; }
    bool supportsBf16() const { return mSupportsBf16; }

private:
    int32_t mDeviceId = -1;
    uint32_t mCoreNum = 0;
    uint64_t mUbSize = 0;
    bool mSupportsFp64 = false;
    bool mSupportsBf16 = false;
};

// Plan 缓存类
class PlanCache {
public:
    PlanCache(uint32_t maxEntries = 64);
    ~PlanCache();

    // 根据操作描述符哈希查询缓存的内核 ID
    uint64_t querySolutionUid(acltensorOperationDescriptor_t desc);

    // 添加缓存条目
    void addCacheLine(uint64_t hashId, uint64_t solutionUid);

    // 调整缓存大小
    void resize(uint32_t numEntries);

    // 持久化
    bool writeFile(const char* fileName);
    bool readFile(const char* fileName, uint32_t* numRead);

private:
    uint32_t mMaxEntries;
    std::unordered_map<uint64_t, uint64_t> mCacheLines;  // hash -> solutionUid
    std::mutex mMutex;
};

} // namespace acltensor

// 句柄结构体
struct acltensorHandle {
    acltensor::AscendDevice device;
    acltensor::PlanCache* planCache = nullptr;
    acltensorLogLevel_t logLevel = ACLTENSOR_LOG_LEVEL_OFF;
};
```

```cpp
// api/core/tensor_descriptor.hpp
#include <vector>
#include <cstddef>

struct acltensorTensorDescriptor {
    acltensorDataType_t dataType;
    uint32_t numModes;
    std::vector<int64_t> lens;      // 各维度长度
    std::vector<int64_t> strides;   // 各维度步长
    uint32_t alignmentRequirement;
    size_t elementSize;             // 单元素字节数

    // 辅助方法
    size_t getTotalElements() const;
    size_t getTotalBytes() const;
    bool isContiguous() const;
};
```

```cpp
// api/core/operation_descriptor.hpp
#include <vector>

// 操作类型
enum class OperationType {
    CONTRACTION,
    REDUCTION,
    PERMUTATION,
    ELEMENTWISE_BINARY,
    ELEMENTWISE_TRINARY
};

struct acltensorOperationDescriptor {
    // 操作类型
    OperationType operationType;

    // 张量描述符
    acltensorTensorDescriptor_t descA = nullptr;
    acltensorTensorDescriptor_t descB = nullptr;
    acltensorTensorDescriptor_t descC = nullptr;
    acltensorTensorDescriptor_t descD = nullptr;

    // 模式数组
    std::vector<int32_t> modeA;
    std::vector<int32_t> modeB;
    std::vector<int32_t> modeC;
    std::vector<int32_t> modeD;

    // 操作符
    acltensorOperator_t opA = ACLTENSOR_OP_IDENTITY;
    acltensorOperator_t opB = ACLTENSOR_OP_IDENTITY;
    acltensorOperator_t opC = ACLTENSOR_OP_IDENTITY;
    acltensorOperator_t opAC = ACLTENSOR_OP_ADD;    // 二元操作
    acltensorOperator_t opAB = ACLTENSOR_OP_ADD;    // 三元操作第一个
    acltensorOperator_t opABC = ACLTENSOR_OP_ADD;   // 三元操作第二个
    acltensorOperator_t opReduce = ACLTENSOR_OP_ADD; // 归约操作

    // 计算精度
    acltensorComputeDescriptor_t descCompute;

    // 用户属性
    uint32_t tag = 0;
    float flops = 0.0f;
    float movedBytes = 0.0f;
    uint32_t paddingLeft = 0;
    uint32_t paddingRight = 0;

    // 计算哈希值（用于缓存）
    uint64_t computeHash() const;
};
```

```cpp
// api/core/plan.hpp

// 内核解决方案基类
class KernelSolution {
public:
    virtual ~KernelSolution() = default;
    virtual uint64_t uid() const = 0;
    virtual size_t workspaceSize() const = 0;
    virtual bool isValid(const acltensorOperationDescriptor_t desc) const = 0;
    virtual acltensorStatus_t execute(
        const acltensorHandle_t handle,
        const acltensorPlan_t plan,
        const void* alpha, const void* A, const void* B,
        const void* beta, const void* C, void* D,
        void* workspace, uint64_t workspaceSize,
        aclrtStream stream) = 0;
};

struct acltensorPlan {
    acltensorOperationDescriptor_t opDesc;
    acltensorPlanPreference_t pref;
    uint64_t requiredWorkspace = 0;

    // 选中的内核解决方案
    std::unique_ptr<KernelSolution> solution;
};
```

---

## 五、各接口实现逻辑设计

### 5.1 句柄管理实现 (Handle Management)

```
acltensorCreate(&handle):
┌─────────────────────────────────────────────────────────────┐
│  1. 分配 handle 内存                                         │
│     handle = new acltensorHandle                            │
│                                                             │
│  2. 初始化 AscendDevice（自动调用构造函数）                   │
│     - aclrtGetDevice(&deviceId)                             │
│     - aclrtGetDeviceCapability() 获取核心数、UB 大小等       │
│     - 检测支持的精度能力 (FP64/BF16)                         │
│                                                             │
│  3. 创建 PlanCache（除非环境变量禁用）                       │
│     handle->planCache = new PlanCache(64)                   │
│                                                             │
│  4. 初始化日志系统                                           │
│     handle->logLevel = ACLTENSOR_LOG_LEVEL_OFF              │
│                                                             │
│  5. 返回 SUCCESS                                            │
└─────────────────────────────────────────────────────────────┘

acltensorDestroy(handle):
┌─────────────────────────────────────────────────────────────┐
│  1. 检查 handle != nullptr                                  │
│                                                             │
│  2. 删除 PlanCache                                          │
│     if (handle->planCache) delete handle->planCache         │
│                                                             │
│  3. 删除 handle                                             │
│     delete handle                                           │
│                                                             │
│  4. 返回 SUCCESS                                            │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 张量描述符实现 (Tensor Descriptor)

```
acltensorCreateTensorDescriptor():
┌─────────────────────────────────────────────────────────────┐
│  1. 参数验证                                                 │
│     - handle != nullptr                                     │
│     - numModes > 0                                          │
│     - lens[i] > 0 for all i                                 │
│                                                             │
│  2. 设备能力检查                                             │
│     - 如果 dataType == FP64 && !device.supportsFp64()       │
│       → return ARCH_MISMATCH                                │
│                                                             │
│  3. 分配描述符内存                                           │
│     desc = new acltensorTensorDescriptor                    │
│                                                             │
│  4. 设置基本属性                                             │
│     desc->dataType = dataType                               │
│     desc->numModes = numModes                               │
│     desc->lens.assign(lens, lens + numModes)                │
│     desc->alignmentRequirement = alignmentRequirement       │
│                                                             │
│  5. 计算步长                                                 │
│     if (strides == nullptr) {                               │
│         // 默认列主序                                        │
│         desc->strides[0] = 1                                │
│         for (i = 1; i < numModes; i++)                      │
│             desc->strides[i] = desc->strides[i-1] * lens[i-1]│
│     } else {                                                │
│         desc->strides.assign(strides, strides + numModes)   │
│     }                                                       │
│                                                             │
│  6. 计算元素大小                                             │
│     desc->elementSize = getElementSize(dataType)            │
│                                                             │
│  7. 返回 SUCCESS                                            │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 张量收缩实现 (Contraction)

```
acltensorCreateContraction():
┌─────────────────────────────────────────────────────────────┐
│  1. 参数验证                                                 │
│     - 所有描述符不为空                                        │
│     - 模式数组维度匹配                                        │
│     - 数据类型兼容性检查                                      │
│                                                             │
│  2. 创建操作描述符                                           │
│     desc = new acltensorOperationDescriptor                 │
│     desc->operationType = CONTRACTION                       │
│                                                             │
│  3. 复制张量描述符和模式                                      │
│     desc->descA/B/C/D = descA/B/C/D                         │
│     desc->modeA/B/C/D = modeA/B/C/D                         │
│     desc->opA/B/C = opA/B/C                                 │
│                                                             │
│  4. 设置计算精度                                             │
│     desc->descCompute = descCompute                         │
│                                                             │
│  5. 验证收缩模式                                             │
│     - 找出收缩维度（在 A 和 B 中都出现但不在 D 中的模式）     │
│     - 验证 A*B 的结果维度与 D 匹配                           │
│                                                             │
│  6. 返回 SUCCESS                                            │
└─────────────────────────────────────────────────────────────┘

acltensorContract():
┌─────────────────────────────────────────────────────────────┐
│  1. 参数验证                                                 │
│     - handle/plan/alpha/beta/A/B/C/D 不为空                 │
│                                                             │
│  2. 获取选中的内核解决方案                                   │
│     solution = plan->solution.get()                         │
│                                                             │
│  3. 检查工作空间大小                                         │
│     if (workspaceSize < plan->requiredWorkspace)            │
│         return INSUFFICIENT_WORKSPACE                       │
│                                                             │
│  4. 执行内核                                                │
│     return solution->execute(                               │
│         handle, plan, alpha, A, B, beta, C, D,              │
│         workspace, workspaceSize, stream)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 Plan 创建和内核选择

```
acltensorCreatePlan():
┌─────────────────────────────────────────────────────────────┐
│  1. 参数验证                                                 │
│                                                             │
│  2. 分配 Plan 内存                                           │
│     plan = new acltensorPlan                                │
│     plan->opDesc = desc                                     │
│                                                             │
│  3. 根据 opDesc->operationType 获取候选内核                  │
│     switch (desc->operationType) {                          │
│         case CONTRACTION:                                   │
│             candidates = getAllContractionKernels()         │
│         case REDUCTION:                                     │
│             candidates = getAllReductionKernels()           │
│         ...                                                 │
│     }                                                       │
│                                                             │
│  4. 过滤候选内核                                             │
│     - 数据类型匹配                                           │
│     - 操作符支持                                             │
│     - 工作空间限制                                           │
│                                                             │
│  5. 查询 PlanCache                                          │
│     if (pref->cacheMode == PEDANTIC && handle->planCache) { │
│         uid = handle->planCache->querySolutionUid(desc)     │
│         if (uid > 0) {                                      │
│             winner = findKernelByUid(candidates, uid)       │
│             goto FOUND                                      │
│         }                                                   │
│     }                                                       │
│                                                             │
│  6. 缓存未命中 → 运行选择算法                                 │
│     switch (pref->algo) {                                   │
│         case DEFAULT:                                       │
│             winner = heuristicSelect(candidates, desc)      │
│         case DEFAULT_PATIENT:                               │
│             winner = bruteForceSelect(candidates, desc)     │
│     }                                                       │
│                                                             │
│  7. 保存选择结果                                             │
│     FOUND:                                                  │
│     plan->solution = std::move(winner)                      │
│     plan->requiredWorkspace = winner->workspaceSize()       │
│                                                             │
│  8. 更新缓存                                                 │
│     if (pref->cacheMode == PEDANTIC)                        │
│         handle->planCache->addCacheLine(hash, winner->uid())│
│                                                             │
│  9. 返回 SUCCESS                                            │
└─────────────────────────────────────────────────────────────┘
```

### 5.5 内核选择算法

```
heuristicSelect(candidates, desc):
┌─────────────────────────────────────────────────────────────┐
│  1. 计算问题特征                                             │
│     - 张量维度 M, N, K                                       │
│     - 内存布局（行主序/列主序）                               │
│     - 数据类型组合                                           │
│                                                             │
│  2. 基于启发式规则排序候选                                   │
│     - 小矩阵 → 小块内核                                      │
│     - 大矩阵 → 大块内核                                      │
│     - 特殊布局 → 专用内核                                    │
│                                                             │
│  3. 返回得分最高的候选                                       │
└─────────────────────────────────────────────────────────────┘

bruteForceSelect(candidates, desc):
┌─────────────────────────────────────────────────────────────┐
│  1. 分配测试内存                                             │
│     aclrtMalloc(&A_test, ...)                               │
│     aclrtMalloc(&B_test, ...)                               │
│     aclrtMalloc(&C_test, ...)                               │
│     aclrtMalloc(&D_test, ...)                               │
│                                                             │
│  2. 遍历所有候选内核，执行性能测试                           │
│     for (kernel in candidates) {                            │
│         // 预热                                             │
│         for (i = 0; i < COLD_RUNS; i++)                     │
│             kernel->execute(...)                            │
│                                                             │
│         // 计时                                             │
│         start = getCurrentTime()                            │
│         for (i = 0; i < HOT_RUNS; i++)                      │
│             kernel->execute(...)                            │
│         end = getCurrentTime()                              │
│         avgTime = (end - start) / HOT_RUNS                  │
│                                                             │
│         // 计算 TFLOPS                                      │
│         tflops = 2 * M * N * K / (avgTime * 1e9)            │
│                                                             │
│         if (tflops > bestTflops) {                          │
│             bestKernel = kernel                             │
│             bestTflops = tflops                             │
│         }                                                   │
│     }                                                       │
│                                                             │
│  3. 释放测试内存                                             │
│     aclrtFree(A_test) ...                                   │
│                                                             │
│  4. 返回最优内核                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 六、AscendC 内核实现示例

### 6.1 矩阵乘法内核（MatMul - 张量收缩的底层实现）

```cpp
// src/matmul/matmul_kernel.cpp
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 16;  // 分块大小

template <typename T>
class MatMulKernel {
public:
    __aicore__ inline MatMulKernel() {}

    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR D,
                                 GM_ADDR tilingGm, TPipe* pipe);
    __aicore__ inline void Process();

private:
    // Tiling 数据
    int64_t M_, N_, K_;
    int64_t blockIdx_;
    int64_t usedCoreNum_;
    int64_t tilesPerCore_;

    // GM 地址
    GlobalTensor<T> gmA_;
    GlobalTensor<T> gmB_;
    GlobalTensor<T> gmC_;
    GlobalTensor<T> gmD_;

    // UB 队列
    TQue<QuePosition::VECIN, 2> inQueueA_;
    TQue<QuePosition::VECIN, 2> inQueueB_;
    TQue<QuePosition::VECOUT, 1> outQueueD_;

    // 累加器
    TBuf<TPosition::LCM> accumBuf_;
};

template <typename T>
__aicore__ inline void MatMulKernel<T>::Process() {
    // 分块矩阵乘法实现
    // D[i,j] = Σ_k A[i,k] * B[k,j] + C[i,j]
    for (int64_t tileIdx = 0; tileIdx < tilesPerCore_; ++tileIdx) {
        // 加载 A 分块到 UB
        // 加载 B 分块到 UB
        // 执行矩阵乘累加
        // 存储 D 分块
    }
}

extern "C" __global__ __aicore__ void matmul_kernel(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR D,
    GM_ADDR workspace, GM_ADDR tilingGm) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    MatMulKernel<float> op;
    op.Init(A, B, C, D, tilingGm, &pipe);
    op.Process();
}
```

### 6.2 归约求和内核（ReduceSum - 张量归约的底层实现）

```cpp
// src/reduce_sum/reduce_sum_kernel.cpp
#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class ReduceSumKernel {
public:
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR D, GM_ADDR tilingGm, TPipe* pipe);
    __aicore__ inline void Process();  // 归约求和

private:
    GlobalTensor<T> gmA_;
    GlobalTensor<T> gmD_;
    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    int64_t reduceDim_;      // 归约维度
    int64_t reduceSize_;     // 归约维度大小
    int64_t outputSize_;     // 输出大小
};
```

---

## 七、典型使用示例

### 7.1 矩阵乘法（张量收缩）

```cpp
#include "cann_ops_tensor.h"
#include <iostream>

// 矩阵乘法: D = α*A*B + β*C
void matmul_example(aclrtStream stream) {
    const int64_t M = 1024, N = 1024, K = 1024;

    // 1. 创建句柄
    acltensorHandle_t handle;
    acltensorCreate(&handle);

    // 2. 创建张量描述符
    int64_t extentA[] = {M, K};
    int64_t extentB[] = {K, N};
    int64_t extentC[] = {M, N};
    int32_t modeA[] = {0, 2};  // m, k
    int32_t modeB[] = {2, 1};  // k, n
    int32_t modeC[] = {0, 1};  // m, n
    int32_t modeD[] = {0, 1};  // m, n

    acltensorTensorDescriptor_t descA, descB, descC, descD;
    acltensorCreateTensorDescriptor(handle, &descA, 2, extentA, NULL, ACLTENSOR_R_32F, 4);
    acltensorCreateTensorDescriptor(handle, &descB, 2, extentB, NULL, ACLTENSOR_R_32F, 4);
    acltensorCreateTensorDescriptor(handle, &descC, 2, extentC, NULL, ACLTENSOR_R_32F, 4);
    acltensorCreateTensorDescriptor(handle, &descD, 2, extentC, NULL, ACLTENSOR_R_32F, 4);

    // 3. 创建收缩操作描述符
    acltensorOperationDescriptor_t opDesc;
    acltensorCreateContraction(handle, &opDesc,
        descA, modeA, ACLTENSOR_OP_IDENTITY,
        descB, modeB, ACLTENSOR_OP_IDENTITY,
        descC, modeC, ACLTENSOR_OP_IDENTITY,
        descD, modeD, ACLTENSOR_COMPUTE_DESC_32F);

    // 4. 创建 Plan Preference
    acltensorPlanPreference_t planPref;
    acltensorCreatePlanPreference(handle, &planPref, ACLTENSOR_ALGO_DEFAULT);

    acltensorCacheMode_t cacheMode = ACLTENSOR_CACHE_MODE_PEDANTIC;
    acltensorPlanPreferenceSetAttribute(handle, planPref,
        ACLTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cacheMode, sizeof(cacheMode));

    // 5. 估算工作空间
    uint64_t workspaceSize;
    acltensorEstimateWorkspaceSize(handle, opDesc, planPref,
        ACLTENSOR_WORKSPACE_DEFAULT, &workspaceSize);

    // 6. 创建 Plan
    acltensorPlan_t plan;
    acltensorCreatePlan(handle, &plan, opDesc, planPref, workspaceSize);

    // 7. 分配设备内存（略）
    void *d_A, *d_B, *d_C, *d_D, *d_workspace;
    // aclrtMalloc(...)

    // 8. 执行收缩
    float alpha = 1.0f, beta = 0.0f;
    acltensorContract(handle, plan, &alpha, d_A, d_B, &beta, d_C, d_D,
        d_workspace, workspaceSize, stream);

    // 9. 清理
    acltensorDestroyPlan(plan);
    acltensorDestroyPlanPreference(planPref);
    acltensorDestroyOperationDescriptor(opDesc);
    acltensorDestroyTensorDescriptor(descA);
    acltensorDestroyTensorDescriptor(descB);
    acltensorDestroyTensorDescriptor(descC);
    acltensorDestroyTensorDescriptor(descD);
    acltensorDestroy(handle);
}
```

### 7.2 张量归约

```cpp
void reduction_example(acltensorHandle_t handle, aclrtStream stream) {
    // 对 3D 张量的第 1 维求和: A(m,n,k) → D(m,k)
    int64_t extentA[] = {128, 256, 64};
    int64_t extentD[] = {128, 64};
    int32_t modeA[] = {0, 1, 2};  // m, n, k
    int32_t modeD[] = {0, 2};     // m, k (n 被归约)

    acltensorTensorDescriptor_t descA, descD;
    acltensorCreateTensorDescriptor(handle, &descA, 3, extentA, NULL, ACLTENSOR_R_32F, 4);
    acltensorCreateTensorDescriptor(handle, &descD, 2, extentD, NULL, ACLTENSOR_R_32F, 4);

    acltensorOperationDescriptor_t opDesc;
    acltensorCreateReduction(handle, &opDesc,
        descA, modeA, ACLTENSOR_OP_IDENTITY,
        descD, modeD, ACLTENSOR_OP_IDENTITY,
        descD, modeD,
        ACLTENSOR_OP_ADD,           // 归约操作符：求和
        ACLTENSOR_COMPUTE_DESC_32F);

    acltensorPlan_t plan;
    acltensorCreatePlan(handle, &plan, opDesc, NULL, 0);

    float alpha = 1.0f, beta = 0.0f;
    acltensorReduce(handle, plan, &alpha, d_A, &beta, d_C, d_D, NULL, 0, stream);

    // ...
}
```

### 7.3 张量置换（转置）

```cpp
void transpose_example(acltensorHandle_t handle, aclrtStream stream) {
    // 矩阵转置: A(m,n) → B(n,m)
    int64_t extentA[] = {1024, 512};
    int64_t extentB[] = {512, 1024};
    int32_t modeA[] = {0, 1};  // m, n
    int32_t modeB[] = {1, 0};  // n, m

    acltensorTensorDescriptor_t descA, descB;
    acltensorCreateTensorDescriptor(handle, &descA, 2, extentA, NULL, ACLTENSOR_R_32F, 4);
    acltensorCreateTensorDescriptor(handle, &descB, 2, extentB, NULL, ACLTENSOR_R_32F, 4);

    acltensorOperationDescriptor_t opDesc;
    acltensorCreatePermutation(handle, &opDesc,
        descA, modeA, ACLTENSOR_OP_IDENTITY,
        descB, modeB, ACLTENSOR_COMPUTE_DESC_32F);

    acltensorPlan_t plan;
    acltensorCreatePlan(handle, &plan, opDesc, NULL, 0);

    float alpha = 1.0f;
    acltensorPermute(handle, plan, &alpha, d_A, d_B, stream);

    // ...
}
```

---

## 八、API 总览表

| 类别 | 函数数量 | 主要接口 |
|------|---------|---------|
| 句柄管理 | 5 | `acltensorCreate`, `acltensorDestroy`, `acltensorHandleResizePlanCache`, `acltensorHandleWritePlanCacheToFile`, `acltensorHandleReadPlanCacheFromFile` |
| 张量描述符 | 2 | `acltensorCreateTensorDescriptor`, `acltensorDestroyTensorDescriptor` |
| 操作描述符 | 9 | `acltensorCreateContraction`, `acltensorCreateReduction`, `acltensorCreatePermutation`, `acltensorCreateElementwiseBinary`, `acltensorCreateElementwiseTrinary`, `acltensorDestroyOperationDescriptor`, `SetAttribute`, `GetAttribute` |
| Plan 管理 | 6 | `acltensorCreatePlanPreference`, `acltensorDestroyPlanPreference`, `acltensorPlanPreferenceSetAttribute`, `acltensorCreatePlan`, `acltensorDestroyPlan`, `acltensorPlanGetAttribute` |
| 执行函数 | 5 | `acltensorContract`, `acltensorReduce`, `acltensorPermute`, `acltensorElementwiseBinaryExecute`, `acltensorElementwiseTrinaryExecute` |
| 辅助工具 | 3 | `acltensorEstimateWorkspaceSize`, `acltensorGetErrorString`, `acltensorGetVersion` |
| 日志系统 | 6 | `acltensorLoggerSetCallback`, `acltensorLoggerSetFile`, `acltensorLoggerOpenFile`, `acltensorLoggerSetLevel`, `acltensorLoggerSetMask`, `acltensorLoggerForceDisable` |

---

## 九、与 hipTensor 的主要差异

| 方面 | hipTensor | ops-tensor |
|------|-----------|------------|
| 底层运行时 | HIP (AMD GPU) | ACL (Ascend NPU) |
| 内核框架 | Composable Kernels | AscendC |
| 并行模型 | SIMT (Thread/Block/Grid) | AI Core (AIV/AIC) |
| 内存层级 | Global/Shared/Registers | GM/UB/L1/L0 |
| 缓存机制 | PlanCache (内核选择结果) | 同样设计 |

---

## 十、后续工作

1. **PlanCache 详细实现** - LRU 淘汰策略、哈希计算、持久化
2. **内核选择算法** - 启发式规则库、性能测试框架
3. **AscendC 内核库** - 各操作类型的内核实现
4. **Tiling 策略** - 自动 tiling 数据计算
5. **单元测试** - API 测试、性能测试
6. **文档完善** - API 文档、使用指南

---

## 附录 A：hipTensor / cuTENSOR / ops-tensor 接口对比

### A.1 句柄管理 (Handle Management)

| 接口功能 | hipTensor | cuTENSOR | ops-tensor |
|---------|-----------|----------|------------|
| 创建句柄 | `hiptensorCreate` | `cutensorCreate` | `acltensorCreate` |
| 销毁句柄 | `hiptensorDestroy` | `cutensorDestroy` | `acltensorDestroy` |
| 调整Plan缓存大小 | `hiptensorHandleResizePlanCache` | `cutensorHandleResizePlanCache` | `acltensorHandleResizePlanCache` |
| 缓存写入文件 | `hiptensorHandleWritePlanCacheToFile` | `cutensorHandleWritePlanCacheToFile` | `acltensorHandleWritePlanCacheToFile` |
| 缓存读取文件 | `hiptensorHandleReadPlanCacheFromFile` | `cutensorHandleReadPlanCacheFromFile` | `acltensorHandleReadPlanCacheFromFile` |

### A.2 张量描述符 (Tensor Descriptor)

| 接口功能 | hipTensor | cuTENSOR | ops-tensor |
|---------|-----------|----------|------------|
| 创建张量描述符 | `hiptensorCreateTensorDescriptor` | `cutensorCreateTensorDescriptor` | `acltensorCreateTensorDescriptor` |
| 销毁张量描述符 | `hiptensorDestroyTensorDescriptor` | `cutensorDestroyTensorDescriptor` | `acltensorDestroyTensorDescriptor` |

### A.3 操作描述符 (Operation Descriptor)

| 接口功能 | hipTensor | cuTENSOR | ops-tensor |
|---------|-----------|----------|------------|
| 创建收缩操作 | `hiptensorCreateContraction` | `cutensorCreateContraction` | `acltensorCreateContraction` |
| 创建归约操作 | `hiptensorCreateReduction` | `cutensorCreateReduction` | `acltensorCreateReduction` |
| 创建置换操作 | `hiptensorCreatePermutation` | `cutensorCreatePermutation` | `acltensorCreatePermutation` |
| 创建二元逐元素 | `hiptensorCreateElementwiseBinary` | `cutensorCreateElementwiseBinary` | `acltensorCreateElementwiseBinary` |
| 创建三元逐元素 | `hiptensorCreateElementwiseTrinary` | `cutensorCreateElementwiseTrinary` | `acltensorCreateElementwiseTrinary` |
| 销毁操作描述符 | `hiptensorDestroyOperationDescriptor` | `cutensorDestroyOperationDescriptor` | `acltensorDestroyOperationDescriptor` |
| 设置属性 | `hiptensorOperationDescriptorSetAttribute` | `cutensorOperationDescriptorSetAttribute` | `acltensorOperationDescriptorSetAttribute` |
| 获取属性 | `hiptensorOperationDescriptorGetAttribute` | `cutensorOperationDescriptorGetAttribute` | `acltensorOperationDescriptorGetAttribute` |

### A.4 Plan 和 PlanPreference

| 接口功能 | hipTensor | cuTENSOR | ops-tensor |
|---------|-----------|----------|------------|
| 创建Plan偏好 | `hiptensorCreatePlanPreference` | `cutensorCreatePlanPreference` | `acltensorCreatePlanPreference` |
| 销毁Plan偏好 | `hiptensorDestroyPlanPreference` | `cutensorDestroyPlanPreference` | `acltensorDestroyPlanPreference` |
| 设置偏好属性 | `hiptensorPlanPreferenceSetAttribute` | `cutensorPlanPreferenceSetAttribute` | `acltensorPlanPreferenceSetAttribute` |
| 估算工作空间 | `hiptensorEstimateWorkspaceSize` | `cutensorEstimateWorkspaceSize` | `acltensorEstimateWorkspaceSize` |
| 创建Plan | `hiptensorCreatePlan` | `cutensorCreatePlan` | `acltensorCreatePlan` |
| 销毁Plan | `hiptensorDestroyPlan` | `cutensorDestroyPlan` | `acltensorDestroyPlan` |
| 获取Plan属性 | `hiptensorPlanGetAttribute` | `cutensorPlanGetAttribute` | `acltensorPlanGetAttribute` |

### A.5 执行函数 (Execution)

| 接口功能 | hipTensor | cuTENSOR | ops-tensor |
|---------|-----------|----------|------------|
| 执行收缩 | `hiptensorContract` | `cutensorContract` | `acltensorContract` |
| 执行归约 | `hiptensorReduce` | `cutensorReduce` | `acltensorReduce` |
| 执行置换 | `hiptensorPermute` | `cutensorPermute` | `acltensorPermute` |
| 执行二元逐元素 | `hiptensorElementwiseBinaryExecute` | `cutensorElementwiseBinaryExecute` | `acltensorElementwiseBinaryExecute` |
| 执行三元逐元素 | `hiptensorElementwiseTrinaryExecute` | `cutensorElementwiseTrinaryExecute` | `acltensorElementwiseTrinaryExecute` |

### A.6 辅助工具 (Utilities)

| 接口功能 | hipTensor | cuTENSOR | ops-tensor |
|---------|-----------|----------|------------|
| 错误字符串 | `hiptensorGetErrorString` | `cutensorGetErrorString` | `acltensorGetErrorString` |
| 获取版本 | `hiptensorGetVersion` | `cutensorGetVersion` | `acltensorGetVersion` |

### A.7 日志系统 (Logging)

| 接口功能 | hipTensor | cuTENSOR | ops-tensor |
|---------|-----------|----------|------------|
| 设置回调 | `hiptensorLoggerSetCallback` | `cutensorLoggerSetCallback` | `acltensorLoggerSetCallback` |
| 设置文件 | `hiptensorLoggerSetFile` | `cutensorLoggerSetFile` | `acltensorLoggerSetFile` |
| 打开文件 | `hiptensorLoggerOpenFile` | `cutensorLoggerOpenFile` | `acltensorLoggerOpenFile` |
| 设置级别 | `hiptensorLoggerSetLevel` | `cutensorLoggerSetLevel` | `acltensorLoggerSetLevel` |
| 设置掩码 | `hiptensorLoggerSetMask` | `cutensorLoggerSetMask` | `acltensorLoggerSetMask` |
| 强制禁用 | `hiptensorLoggerForceDisable` | `cutensorLoggerForceDisable` | `acltensorLoggerForceDisable` |

### A.8 接口总数

| 类别 | 数量 |
|------|------|
| 句柄管理 | 5 |
| 张量描述符 | 2 |
| 操作描述符 | 8 |
| Plan管理 | 7 |
| 执行函数 | 5 |
| 辅助工具 | 2 |
| 日志系统 | 6 |
| **总计** | **35** |

### A.9 三者关系

```
cuTENSOR (NVIDIA)  ←──API高度兼容──→  hipTensor (AMD)
         │                                    │
         │         相同设计模式                │
         └──────────→ ops-tensor (Ascend) ←───┘

命名规则：
- cuTENSOR:  cutensorXxx
- hipTensor: hiptensorXxx
- ops-tensor: acltensorXxx  (遵循 CANN acl 命名风格)
```

---

## 附录 B：代码量预估

### B.1 预估依据

基于 hipTensor 开源项目（约 15K 行核心代码）及典型张量运算库的代码规模进行预估。

### B.2 分模块代码量预估

| 模块 | 目录 | 主要内容 | 预估代码量 (KLOC) |
|------|------|----------|------------------|
| **include/** | 公共头文件 | 类型定义、API声明 | 0.5 |
| **api/core/** | 核心框架 | Handle、Descriptor、Plan、PlanCache | 2.0 |
| **api/elementwise/** | Elementwise框架 | Binary/Trinary 分发逻辑 | 1.0 |
| **api/contraction/** | Contraction框架 | 收缩操作分发、解决方案选择 | 1.5 |
| **api/reduction/** | Reduction框架 | 归约操作分发 | 0.8 |
| **api/permutation/** | Permutation框架 | 置换操作分发 | 0.5 |
| **api/utils/** | 工具函数 | Tiling计算、内存管理 | 0.5 |
| **src/add/** | Add算子 | Host + Kernel | 0.3 |
| **src/mul/** | Mul算子 | Host + Kernel | 0.3 |
| **src/sub/** | Sub算子 | Host + Kernel | 0.3 |
| **src/div/** | Div算子 | Host + Kernel | 0.3 |
| **src/max/** | Max算子 | Host + Kernel | 0.3 |
| **src/min/** | Min算子 | Host + Kernel | 0.3 |
| **src/matmul/** | MatMul算子 | Host + Kernel（多种分块策略） | 1.5 |
| **src/reduce_sum/** | ReduceSum算子 | Host + Kernel | 0.5 |
| **src/reduce_max/** | ReduceMax算子 | Host + Kernel | 0.5 |
| **src/reduce_min/** | ReduceMin算子 | Host + Kernel | 0.5 |
| **src/transpose/** | Transpose算子 | Host + Kernel | 0.5 |
| **tests/** | 测试代码 | 单元测试、性能测试 | 1.5 |
| **CMake/构建** | 构建脚本 | CMakeLists、配置文件 | 0.3 |

### B.3 阶段性代码量汇总

| 阶段 | 内容 | 预估代码量 (KLOC) |
|------|------|------------------|
| **阶段一：核心框架** | include/ + api/core/ + api/utils/ | 3.0 K |
| **阶段二：Elementwise** | api/elementwise/ + 基础算子(add/mul/sub/div) | 2.2 K |
| **阶段三：Contraction** | api/contraction/ + matmul算子 | 3.0 K |
| **阶段四：Reduction** | api/reduction/ + reduce算子 | 1.8 K |
| **阶段五：Permutation** | api/permutation/ + transpose算子 | 1.0 K |
| **阶段六：测试与文档** | tests/ + 文档 | 2.0 K |

### B.4 总代码量预估

| 分类 | 预估代码量 (KLOC) |
|------|------------------|
| **框架层 (api/)** | 6.3 K |
| **算子层 (src/)** | 5.0 K |
| **头文件 (include/)** | 0.5 K |
| **测试代码 (tests/)** | 1.5 K |
| **构建脚本** | 0.3 K |
| **总计** | **~13.6 K** |

### B.5 与参考项目对比

| 项目 | 代码量 | 说明 |
|------|--------|------|
| hipTensor (AMD) | ~15 K | 完整实现，包含多种数据类型 |
| cuTENSOR (NVIDIA) | 闭源 | 仅 API 参考 |
| **ops-tensor (预估)** | **~13.6 K** | 初期版本，支持核心数据类型 |

### B.6 备注

1. **初期版本**：优先支持 FP32/FP16 数据类型，后续扩展 BF16/INT8/FP64
2. **算子扩展**：每增加一个新算子约增加 0.3-0.5 KLOC
3. **性能优化**：MatMul 多种分块策略可能增加 1-2 KLOC
4. **数据类型扩展**：每增加一个数据类型约增加 20% 的 Kernel 代码

---

## 附录 C：Elementwise Binary 核心结构体设计

本章节详细描述 ops-tensor Elementwise Binary 操作所需的核心结构体设计。

### C.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Elementwise Binary 调用链                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  acltensorCreate(&handle)                                               │
│        │                                                                │
│        ▼                                                                │
│  acltensorCreateTensorDescriptor(handle, &descA, ...)  ──┐              │
│  acltensorCreateTensorDescriptor(handle, &descC, ...)  ──┼── 元数据     │
│  acltensorCreateTensorDescriptor(handle, &descD, ...)  ──┘   (无数据)   │
│        │                                                                │
│        ▼                                                                │
│  acltensorCreateElementwiseBinary(handle, &opDesc,                      │
│      descA, modeA, opA,                                                 │
│      descC, modeC, opC,                                                 │
│      descD, modeD, opAC, descCompute)    ← 操作描述                      │
│        │                                                                │
│        ▼                                                                │
│  acltensorCreatePlanPreference(handle, &pref, algo, jitMode)            │
│        │                                                                │
│        ▼                                                                │
│  acltensorCreatePlan(handle, &plan, opDesc, pref, workspaceLimit)       │
│        │                                     ↑                          │
│        │                              选择最优内核                        │
│        ▼                                                                │
│  acltensorElementwiseBinaryExecute(handle, plan,                        │
│      &alpha, d_A, &gamma, d_C, d_D, stream)  ← 执行                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### C.2 Handle 结构体设计

**文件位置**: `api/core/handle.hpp`

```cpp
#include <mutex>
#include <memory>
#include "acl/acl.h"

namespace acltensor {

// Ascend 设备信息
class AscendDevice {
public:
    AscendDevice();
    ~AscendDevice() = default;

    // 设备信息查询
    int32_t     getDeviceId()    const { return mDeviceId; }
    uint32_t    getCoreNum()     const { return mCoreNum; }      // AIV 核数
    uint32_t    getCoreNumAic()  const { return mCoreNumAic; }   // AIC 核数
    uint64_t    getUbSize()      const { return mUbSize; }       // UB 大小
    const char* getSoCName()     const { return mSoCName; }      // SoC 名称

    // 能力查询
    bool supportsFp16()    const { return mSupportsFp16; }
    bool supportsFp32()    const { return true; }  // 始终支持
    bool supportsBf16()    const { return mSupportsBf16; }
    bool supportsInt8()    const { return mSupportsInt8; }

private:
    int32_t     mDeviceId      = -1;
    uint32_t    mCoreNum       = 0;
    uint32_t    mCoreNumAic    = 0;
    uint64_t    mUbSize        = 0;
    char        mSoCName[64]   = {0};

    bool        mSupportsFp16  = false;
    bool        mSupportsBf16  = false;
    bool        mSupportsInt8  = false;
};

// Plan 缓存（LRU 策略）
class PlanCache {
public:
    explicit PlanCache(uint32_t maxEntries = 64);
    ~PlanCache();

    // 根据操作描述符哈希查询缓存的内核 ID
    uint64_t querySolutionUid(uint64_t hashId);

    // 添加缓存条目
    void addCacheLine(uint64_t hashId, uint64_t solutionUid);

    // 管理接口
    void resize(uint32_t numEntries);
    void clear();
    uint32_t size() const { return mCacheLines.size(); }

    // 持久化（可选）
    bool writeFile(const char* fileName);
    bool readFile(const char* fileName, uint32_t* numRead);

private:
    uint32_t mMaxEntries;

    // 哈希表：操作哈希 → 内核 ID
    std::unordered_map<uint64_t, uint64_t> mCacheLines;

    // LRU 优先队列（用于淘汰）
    std::list<uint64_t> mLRUList;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator> mLRUMap;

    std::mutex mMutex;
};

} // namespace acltensor

// 对外句柄结构体
struct acltensorHandle {
    acltensor::AscendDevice  device;              // 设备信息
    acltensor::PlanCache*    planCache = nullptr; // Plan 缓存
    acltensorLogLevel_t      logLevel = ACLTENSOR_LOG_LEVEL_OFF;

    // 可选：当前 stream（某些操作可能需要）
    aclrtStream              currentStream = nullptr;
};
```

**Handle 成员职责**:

| 成员 | 用途 | 生命周期 |
|------|------|----------|
| `AscendDevice device` | 设备信息，只读 | 随 Handle 创建/销毁 |
| `PlanCache* planCache` | 内核选择结果缓存 | 可选，可通过环境变量禁用 |
| `logLevel` | 日志级别 | 用户可配置 |

### C.3 TensorDescriptor 结构体设计

**文件位置**: `api/core/tensor_descriptor.hpp`

```cpp
#include <vector>
#include <cstdint>

struct acltensorTensorDescriptor {
    // 基本属性
    acltensorDataType_t  dataType;              // 数据类型
    uint32_t             numModes;              // 维度数量
    std::vector<int64_t> lens;                  // 各维度长度 [M, N, K, ...]
    std::vector<int64_t> strides;               // 各维度步长
    uint32_t             alignmentRequirement;  // 内存对齐要求（字节）

    // 派生属性（创建时计算，避免重复）
    size_t               elementSize;           // 单元素字节数
    size_t               totalElements;         // 总元素数
    size_t               totalBytes;            // 总字节数
    bool                 isContiguous;          // 是否连续内存

    // 辅助方法
    size_t getElementSize(acltensorDataType_t type) const;
    void   computeDerivedAttributes();
    bool   checkContiguous() const;
};
```

**派生属性计算逻辑**:

```cpp
void acltensorTensorDescriptor::computeDerivedAttributes() {
    // 1. 计算元素大小
    elementSize = getElementSize(dataType);

    // 2. 计算总元素数
    totalElements = 1;
    for (auto len : lens) {
        totalElements *= len;
    }

    // 3. 计算总字节数
    totalBytes = totalElements * elementSize;

    // 4. 检查是否连续
    isContiguous = checkContiguous();
}

bool acltensorTensorDescriptor::checkContiguous() const {
    if (strides.empty()) return true;

    // 检查是否为列主序连续
    int64_t expectedStride = 1;
    for (uint32_t i = 0; i < numModes; ++i) {
        if (strides[i] != expectedStride) return false;
        expectedStride *= lens[i];
    }
    return true;
}
```

### C.4 OperationDescriptor 结构体设计

**文件位置**: `api/core/operation_descriptor.hpp`

```cpp
#include <vector>

// 操作类型枚举
enum class OperationType : uint32_t {
    CONTRACTION        = 0,
    REDUCTION          = 1,
    PERMUTATION        = 2,
    ELEMENTWISE_BINARY = 3,
    ELEMENTWISE_TRINARY = 4,
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
    acltensorOperator_t opA  = ACLTENSOR_OP_IDENTITY;  // A 的预处理
    acltensorOperator_t opC  = ACLTENSOR_OP_IDENTITY;  // C 的预处理
    acltensorOperator_t opAC = ACLTENSOR_OP_ADD;       // 二元操作 (ADD/MUL/MAX/MIN)

    // Elementwise Trinary 专用
    acltensorOperator_t opB   = ACLTENSOR_OP_IDENTITY;
    acltensorOperator_t opAB  = ACLTENSOR_OP_ADD;    // 第一个二元操作
    acltensorOperator_t opABC = ACLTENSOR_OP_ADD;    // 第二个二元操作

    // Reduction 专用
    acltensorOperator_t opReduce = ACLTENSOR_OP_ADD;

    // ========== 计算精度 ==========
    acltensorComputeDescriptor_t descCompute;

    // ========== 用户属性 ==========
    uint32_t tag          = 0;      // 用户标签（参与哈希）
    float    flops        = 0.0f;   // 计算量估算
    float    movedBytes   = 0.0f;   // 数据移动量

    // ========== 派生属性（创建时计算）==========
    acltensorDataType_t scalarType;      // α/γ 标量类型
    uint64_t            hashValue;       // 用于缓存的哈希值

    // ========== 辅助方法 ==========
    uint64_t computeHash() const;
    void     deriveScalarType();
};
```

**哈希计算逻辑**（用于 PlanCache）:

```cpp
uint64_t acltensorOperationDescriptor::computeHash() const {
    uint64_t hash = 0;

    // 操作类型
    hash = hashCombine(hash, static_cast<uint32_t>(operationType));

    // 用户标签
    hash = hashCombine(hash, tag);

    // 数据类型
    if (descA) hash = hashCombine(hash, static_cast<uint32_t>(descA->dataType));
    if (descC) hash = hashCombine(hash, static_cast<uint32_t>(descC->dataType));
    if (descD) hash = hashCombine(hash, static_cast<uint32_t>(descD->dataType));
    hash = hashCombine(hash, static_cast<uint32_t>(descCompute));

    // 操作符
    hash = hashCombine(hash, static_cast<uint32_t>(opA));
    hash = hashCombine(hash, static_cast<uint32_t>(opC));
    hash = hashCombine(hash, static_cast<uint32_t>(opAC));

    // 维度信息
    for (auto m : modeA) hash = hashCombine(hash, m);
    for (auto m : modeC) hash = hashCombine(hash, m);
    for (auto m : modeD) hash = hashCombine(hash, m);

    // 形状信息
    if (descA) {
        for (auto len : descA->lens) hash = hashCombine(hash, len);
    }
    if (descC) {
        for (auto len : descC->lens) hash = hashCombine(hash, len);
    }

    return hash;
}
```

### C.5 PlanPreference 结构体设计

**文件位置**: `api/core/plan_preference.hpp`

```cpp
// 自动调优模式
enum class acltensorAutotuneMode_t : uint32_t {
    NONE        = 0,  // 不自动调优
    INCREMENTAL = 1,  // 增量调优
};

// 缓存模式
enum class acltensorCacheMode_t : uint32_t {
    NONE     = 0,  // 不使用缓存
    PEDANTIC = 1,  // 使用缓存（推荐）
};

// JIT 模式
enum class acltensorJitMode_t : uint32_t {
    NONE    = 0,  // 不使用 JIT
    DEFAULT = 1,  // 默认 JIT 模式
};

// 内核秩（用于 Contraction）
enum class acltensorKernelRank_t : uint32_t {
    RANK_0 = 0,
    RANK_1 = 1,
    RANK_2 = 2,
};

struct acltensorPlanPreference {
    // 选择策略
    acltensorAlgo_t         algo            = ACLTENSOR_ALGO_DEFAULT;

    // 缓存与调优
    acltensorCacheMode_t    cacheMode       = acltensorCacheMode_t::PEDANTIC;
    acltensorAutotuneMode_t autotuneMode    = acltensorAutotuneMode_t::NONE;

    // 内核配置
    acltensorJitMode_t      jitMode         = acltensorJitMode_t::NONE;
    acltensorKernelRank_t   kernelRank      = acltensorKernelRank_t::RANK_2;

    // 增量计数（用于 INCREMENTAL 调优）
    uint32_t                incrementalCount = 0;

    // ========== 内部状态（CreatePlan 时填充）==========
    std::vector<void*>      candidates;     // 候选内核列表
    void*                   selectedSolution = nullptr;  // 选中的内核
};
```

### C.6 Plan 结构体设计

**文件位置**: `api/core/plan.hpp`

```cpp
#include <memory>

// 内核解决方案基类（抽象接口）
class KernelSolution {
public:
    virtual ~KernelSolution() = default;

    virtual uint64_t uid() const = 0;                    // 唯一标识符
    virtual size_t   workspaceSize() const = 0;          // 所需工作空间
    virtual bool     isValid() const = 0;                // 是否有效

    // 执行接口
    virtual acltensorStatus_t execute(
        const acltensorHandle_t handle,
        const void* alpha, const void* A,
        const void* gamma, const void* C,
        void* D,
        aclrtStream stream) = 0;
};

// Elementwise Binary 专用解决方案
class ElementwiseBinarySolution : public KernelSolution {
public:
    ElementwiseBinarySolution(
        uint64_t uid,
        acltensorOperator_t opAC,
        acltensorDataType_t dataType,
        KernelExecuteFunc  executeFunc);

    uint64_t uid() const override { return mUid; }
    size_t   workspaceSize() const override { return 0; }  // Elementwise 通常不需要
    bool     isValid() const override { return mExecuteFunc != nullptr; }

    acltensorStatus_t execute(
        const acltensorHandle_t handle,
        const void* alpha, const void* A,
        const void* gamma, const void* C,
        void* D,
        aclrtStream stream) override;

private:
    uint64_t              mUid;
    acltensorOperator_t   mOpAC;
    acltensorDataType_t   mDataType;
    KernelExecuteFunc     mExecuteFunc;  // 函数指针，指向具体算子
};

// 函数指针类型（指向 src/{op}/ 目录下的算子）
typedef acltensorStatus_t (*KernelExecuteFunc)(
    const void* A, const void* C, void* D,
    int64_t elemNum,
    const void* alpha, const void* gamma,
    aclrtStream stream
);

struct acltensorPlan {
    // 关联的操作描述符
    acltensorOperationDescriptor_t opDesc;

    // 关联的偏好设置（包含选中的内核）
    acltensorPlanPreference_t pref;

    // 所需工作空间大小
    uint64_t requiredWorkspace = 0;

    // 选中的内核解决方案
    std::unique_ptr<KernelSolution> solution;
};
```

### C.7 结构体关系图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           结构体关系                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  acltensorHandle                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  AscendDevice device;                                            │   │
│  │  PlanCache*   planCache;  ─────────────────────┐                │   │
│  └────────────────────────────────────────────────┼────────────────┘   │
│                                                   │                    │
│                                                   ▼                    │
│  acltensorOperationDescriptor              ┌──────────────────┐        │
│  ┌─────────────────────────────────────┐   │   PlanCache      │        │
│  │  OperationType = ELEMENTWISE_BINARY │   │  ┌────────────┐  │        │
│  │                                     │   │  │ hash → uid │  │        │
│  │  descA ──► TensorDescriptor         │   │  │ hash → uid │  │        │
│  │  descC ──► TensorDescriptor         │   │  │ ...        │  │        │
│  │  descD ──► TensorDescriptor         │   │  └────────────┘  │        │
│  │                                     │   └──────────────────┘        │
│  │  modeA, modeC, modeD                │                               │
│  │  opA, opC, opAC                     │                               │
│  │  descCompute                        │                               │
│  │  hashValue ────────────────────────────────► 用于查询 PlanCache    │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  acltensorPlanPreference                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  algo = DEFAULT                                                  │   │
│  │  cacheMode = PEDANTIC                                            │   │
│  │  candidates = [Solution1, Solution2, ...]  ← 候选内核            │   │
│  │  selectedSolution = Solution3  ◄─────────── 选中                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  acltensorPlan                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  opDesc = (关联操作描述符)                                        │   │
│  │  pref   = (关联偏好设置，含选中内核)                              │   │
│  │  requiredWorkspace = 0                                           │   │
│  │  solution = unique_ptr<KernelSolution>                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### C.8 各结构体职责总结

| 结构体 | 职责 | 创建时机 | 销毁时机 |
|--------|------|----------|----------|
| `Handle` | 库上下文，管理设备信息和缓存 | 程序初始化时一次 | 程序退出时 |
| `TensorDescriptor` | 描述张量形状、类型、布局 | 每种形状一次 | 不再使用时 |
| `OperationDescriptor` | 描述操作语义 | 每种操作配置一次 | 不再使用时 |
| `PlanPreference` | 内核选择偏好 | 可选，复用 | 不再使用时 |
| `Plan` | 执行计划，包含选中内核 | 一次（可能耗时） | 不再使用时 |

### C.9 核心设计原则

1. **一次创建，多次执行**：Plan 创建后可复用于多次 Execute
2. **缓存加速**：相同配置的 Plan 从缓存中获取，避免重复选择内核
3. **分层解耦**：框架层（api/elementwise/）→ 算子层（src/{op}/）
4. **延迟绑定**：α/γ 等标量在 Execute 时传入，而非 Create 时
