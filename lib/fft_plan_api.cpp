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
 * @brief 创建空的 FFT Plan
 *
 * 创建一个未初始化的 Plan，需要后续调用 aclfftMakePlan1d/2d/3d 来初始化
 */
aclfftResult aclfftCreate(aclfftHandle* plan) {
    // 参数验证
    ACLFFT_CHECK_NULL(plan);

    // 创建 Plan 对象
    aclfftHandle_t* new_plan = new (std::nothrow) aclfftHandle_t();
    if (new_plan == nullptr) {
        return ACLFFT_ALLOC_FAILED;
    }

    *plan = new_plan;
    return ACLFFT_SUCCESS;
}

/**
 * @brief 创建并初始化 1D FFT Plan
 *
 * @param plan 输出参数：返回创建的 Plan 句柄
 * @param nx X 维度大小
 * @param type 变换类型
 * @param batch 批量大小
 * @return aclfftResult 错误码
 */
aclfftResult aclfftPlan1d(aclfftHandle* plan, int nx, aclfftType type, int batch) {
    // 参数验证
    ACLFFT_CHECK_NULL(plan);
    ACLFFT_CHECK_PARAM(nx > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(batch > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(type >= ACLFFT_C2C && type <= ACLFFT_Z2D, ACLFFT_INVALID_TYPE);

    // 创建 Plan
    aclfftResult res = aclfftCreate(plan);
    if (res != ACLFFT_SUCCESS) {
        return res;
    }

    // 初始化 Plan
    size_t workSize;
    res = aclfftMakePlan1d(*plan, nx, type, batch, &workSize);

    // 如果初始化失败，清理 Plan
    if (res != ACLFFT_SUCCESS) {
        aclfftDestroy(*plan);
        *plan = nullptr;
    }

    return res;
}

/**
 * @brief 创建并初始化 2D FFT Plan
 */
aclfftResult aclfftPlan2d(aclfftHandle* plan, int nx, int ny, aclfftType type) {
    // 参数验证
    ACLFFT_CHECK_NULL(plan);
    ACLFFT_CHECK_PARAM(nx > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(ny > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(type >= ACLFFT_C2C && type <= ACLFFT_Z2D, ACLFFT_INVALID_TYPE);

    // 创建 Plan
    aclfftResult res = aclfftCreate(plan);
    if (res != ACLFFT_SUCCESS) {
        return res;
    }

    // 初始化 Plan
    size_t workSize;
    res = aclfftMakePlan2d(*plan, nx, ny, type, &workSize);

    // 如果初始化失败，清理 Plan
    if (res != ACLFFT_SUCCESS) {
        aclfftDestroy(*plan);
        *plan = nullptr;
    }

    return res;
}

/**
 * @brief 创建并初始化 3D FFT Plan
 */
aclfftResult aclfftPlan3d(aclfftHandle* plan, int nx, int ny, int nz, aclfftType type) {
    // 参数验证
    ACLFFT_CHECK_NULL(plan);
    ACLFFT_CHECK_PARAM(nx > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(ny > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(nz > 0, ACLFFT_INVALID_SIZE);
    ACLFFT_CHECK_PARAM(type >= ACLFFT_C2C && type <= ACLFFT_Z2D, ACLFFT_INVALID_TYPE);

    // 创建 Plan
    aclfftResult res = aclfftCreate(plan);
    if (res != ACLFFT_SUCCESS) {
        return res;
    }

    // 初始化 Plan
    size_t workSize;
    res = aclfftMakePlan3d(*plan, nx, ny, nz, type, &workSize);

    // 如果初始化失败，清理 Plan
    if (res != ACLFFT_SUCCESS) {
        aclfftDestroy(*plan);
        *plan = nullptr;
    }

    return res;
}
