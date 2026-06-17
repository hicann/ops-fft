/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "irfft1_d.h"
#include "fft_exec_helper.h"

extern "C" {
aclfftResult aclfftExecC2R_1D(aclfftHandle plan,
                              aclfftComplex* idata,
                              aclfftReal* odata) {
    aclfftHandle_t* impl = plan;
    ACLFFT_CHECK_PARAM(impl->rank == 1, ACLFFT_INVALID_VALUE);

    const int64_t n = static_cast<int64_t>(impl->lengths[0]);
    const uint32_t batch = impl->batch;
    int32_t irfft_norm = impl->normMode + 1;
    int isForward = 0;

    aclError err;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    auto socVersion = ascendcPlatform->GetSocVersion();

    static constexpr int64_t K_N_FFT_1024 = 1024;

    if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        if (n <= K_N_FFT_1024) {
            err = aclfftIrfft1DDft(reinterpret_cast<float*>(idata),
                                reinterpret_cast<float*>(odata),
                                static_cast<uint32_t>(n), irfft_norm, batch, isForward, impl->stream);
        } else {
            std::cout << "[ops-fft] C2R: n=" << n << " would use FFT path (n > 1024), not implemented" << std::endl;
            return ACLFFT_NOT_IMPLEMENTED;
        }
    } else {
        std::cerr << "  [ERROR] Unsupported SoC: " << SocVersionToString(socVersion)
                  << ", expected Ascend910B" << std::endl;
        err = ACLFFT_EXEC_FAILED;
    }

    return (err == ACL_SUCCESS) ? ACLFFT_SUCCESS : ACLFFT_EXEC_FAILED;
}
} // extern "C"
