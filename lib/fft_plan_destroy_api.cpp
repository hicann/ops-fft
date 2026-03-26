/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include "cann_ops_fft.h"
#include "fft_handle_impl.h"
#include "fft_error.h"

/**
 * @brief 销毁 FFT Plan
 *
 * 释放 Plan 及其相关资源
 *
 * @param plan Plan 句柄
 * @return aclfftResult 错误码
 */
aclfftResult aclfftDestroy(aclfftHandle plan) {
    aclfftHandle_t* impl = plan;

    // 参数验证
    // 注意：允许销毁未初始化的 Plan
    ACLFFT_CHECK_NULL(impl);

    // 防止重复销毁
    if (impl->is_destroyed) {
        return ACLFFT_INVALID_PLAN;
    }

    if (impl->has_operator_state && impl->operator_state != nullptr) {}

    // 标记为已销毁
    impl->is_destroyed = true;

    // 释放 Plan 对象
    delete impl;

    return ACLFFT_SUCCESS;
}
