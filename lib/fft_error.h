/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT_ERROR_H
#define FFT_ERROR_H

#include "cann_ops_fft.h"

// ============================================================================
// 错误处理宏
// ============================================================================

/**
 * @brief 参数验证宏
 *
 * 如果条件为 false，返回指定的错误码
 */
#define ACLFFT_CHECK_PARAM(condition, error_code) \
    do { \
        if (!(condition)) { \
            return (error_code); \
        } \
    } while(0)

/**
 * @brief ACL 错误检查宏
 *
 * 如果 ACL 调用失败，返回 ACLFFT_EXEC_FAILED
 */
#define ACLFFT_CHECK_ACL(call) \
    do { \
        aclError _err = (call); \
        if (_err != ACL_SUCCESS) { \
            return ACLFFT_EXEC_FAILED; \
        } \
    } while(0)

/**
 * @brief 空指针检查宏
 */
#define ACLFFT_CHECK_NULL(ptr) \
    ACLFFT_CHECK_PARAM((ptr) != nullptr, ACLFFT_INVALID_VALUE)

/**
 * @brief Plan 有效性检查宏
 */
#define ACLFFT_CHECK_PLAN(plan) \
    do { \
        ACLFFT_CHECK_NULL(plan); \
        ACLFFT_CHECK_PARAM(!(plan)->is_destroyed, ACLFFT_INVALID_PLAN); \
    } while(0)

/**
 * @brief Plan 初始化状态检查宏
 */
#define ACLFFT_CHECK_PLAN_INITIALIZED(plan) \
    do { \
        ACLFFT_CHECK_PLAN(plan); \
        ACLFFT_CHECK_PARAM((plan)->is_initialized, ACLFFT_INVALID_PLAN); \
    } while(0)

/**
 * @brief Plan 未初始化检查宏
 */
#define ACLFFT_CHECK_PLAN_NOT_INITIALIZED(plan) \
    do { \
        ACLFFT_CHECK_PLAN(plan); \
        ACLFFT_CHECK_PARAM(!(plan)->is_initialized, ACLFFT_INVALID_VALUE); \
    } while(0)

#endif // FFT_ERROR_H
