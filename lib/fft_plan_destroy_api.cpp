/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
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

    // 说明：原 is_destroyed 防重复销毁检查本身构成 use-after-free——对象在首次
    // 销毁时即被 delete，标志随对象一同释放，二次调用读取的是已释放内存。
    // 重复销毁应由调用方保证（内部调用方在销毁后均置 *plan=nullptr），
    // 对已销毁句柄的再次传入属未定义行为，此处不再解引用已释放对象（issue #60）。

    // 释放 Plan 对象
    delete impl;

    return ACLFFT_SUCCESS;
}
