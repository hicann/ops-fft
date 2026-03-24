/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_ops_fft.h"
#include "fft_handle_impl.h"
#include "fft_error.h"

/**
 * @brief 设置 FFT Plan 的执行流
 *
 * @param plan Plan 句柄
 * @param stream ACL 流句柄
 * @return aclfftResult 错误码
 */
aclfftResult aclfftSetStream(aclfftHandle plan, aclrtStream stream) {
    aclfftHandle_t* impl = plan;

    // 参数验证
    ACLFFT_CHECK_PLAN(impl);

    // 设置流
    impl->stream = stream;

    return ACLFFT_SUCCESS;
}
