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
#include "rfft1_d.h"  // rfft1_d 算子接口

/**
 * @brief 执行 C2C FFT
 *
 * @param plan Plan 句柄
 * @param idata 输入数据
 * @param odata 输出数据
 * @param direction 变换方向（ACLFFT_FORWARD 或 ACLFFT_BACKWARD）
 * @return aclfftResult 错误码
 */
aclfftResult aclfftExecC2C(aclfftHandle plan,
                           aclfftComplex* idata,
                           aclfftComplex* odata,
                           int direction) {
    return ACLFFT_NOT_IMPLEMENTED;
}

/**
 * @brief 执行 R2C FFT
 *
 * 直接调用 rfft1_d 算子（aclfftRfft1D）
 */
aclfftResult aclfftExecR2C(aclfftHandle plan,
                           aclfftReal* idata,
                           aclfftComplex* odata) {
    aclfftHandle_t* impl = plan;

    // 参数验证
    ACLFFT_CHECK_PLAN_INITIALIZED(impl);
    ACLFFT_CHECK_NULL(idata);
    ACLFFT_CHECK_NULL(odata);
    ACLFFT_CHECK_PARAM(impl->type == ACLFFT_R2C, ACLFFT_INVALID_TYPE);
    ACLFFT_CHECK_PARAM(impl->rank == 1, ACLFFT_INVALID_VALUE);

    // 直接调用 rfft1_d 算子
    const uint32_t n = impl->lengths[0];
    const uint32_t batch = impl->batch;

    // 归一化模式映射：内部使用 0-based，算子使用 1-based
    int32_t rfft_norm = impl->normMode + 1;

    aclError err = aclfftRfft1D(reinterpret_cast<float*>(idata),
                                reinterpret_cast<float*>(odata),
                                n, rfft_norm, batch, impl->stream);

    return (err == ACL_SUCCESS) ? ACLFFT_SUCCESS : ACLFFT_EXEC_FAILED;
}

/**
 * @brief 执行 C2R FFT
 *
 * 占位实现，返回 ACLFFT_NOT_IMPLEMENTED
 */
aclfftResult aclfftExecC2R(aclfftHandle plan,
                           aclfftComplex* idata,
                           aclfftReal* odata) {
    return ACLFFT_NOT_IMPLEMENTED;
}

/**
 * @brief 执行 Z2Z FFT
 *
 * 占位实现，返回 ACLFFT_NOT_IMPLEMENTED
 */
aclfftResult aclfftExecZ2Z(aclfftHandle plan,
                           aclfftDoubleComplex* idata,
                           aclfftDoubleComplex* odata,
                           int direction) {
    return ACLFFT_NOT_IMPLEMENTED;
}

/**
 * @brief 执行 D2Z FFT
 *
 * 占位实现，返回 ACLFFT_NOT_IMPLEMENTED
 */
aclfftResult aclfftExecD2Z(aclfftHandle plan,
                           aclfftDoubleReal* idata,
                           aclfftDoubleComplex* odata) {
    return ACLFFT_NOT_IMPLEMENTED;
}

/**
 * @brief 执行 Z2D FFT
 *
 * 占位实现，返回 ACLFFT_NOT_IMPLEMENTED
 */
aclfftResult aclfftExecZ2D(aclfftHandle plan,
                           aclfftDoubleComplex* idata,
                           aclfftDoubleReal* odata) {
    return ACLFFT_NOT_IMPLEMENTED;
}
