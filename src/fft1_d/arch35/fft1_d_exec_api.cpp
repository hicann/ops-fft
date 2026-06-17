/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fft1_d.h"
#include "fft_exec_helper.h"

extern "C" {
aclfftResult aclfftExecC2C_1D(aclfftHandle plan,
                           aclfftComplex* idata,
                           aclfftComplex* odata,
                           int direction) {
    aclfftHandle_t* impl = plan;
    ACLFFT_CHECK_PARAM(impl->rank == 1, ACLFFT_INVALID_VALUE);

    const uint32_t n = impl->lengths[0];
    const uint32_t batch = impl->batch;
    int32_t fft_norm = impl->normMode + 1;
    int isForward = direction == ACLFFT_FORWARD ? 1 : 0;
    aclError err;

    std::vector<int64_t> factors = orderedFactorize(impl->lengths[0]);
    std::vector<int64_t> uniques = deDuplicates(factors);

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    auto socVersion = ascendcPlatform->GetSocVersion();

    if (socVersion == platform_ascendc::SocVersion::ASCEND950) {
        int radix = ChooseRadix(impl->type, uniques);
        if (n > K_N_FFT_256 && (radix == K_RADIX_2 || radix == K_RADIX_MIX)) {
            err = aclfftFft1DC2C(reinterpret_cast<float*>(idata),
                                 reinterpret_cast<float*>(odata),
                                 n, fft_norm, batch, isForward, impl->stream);
        } else {
            std::cerr << "[ops-fft] C2C arch35: n=" << n << " radix=" << radix << " not implemented" << std::endl;
            return ACLFFT_NOT_IMPLEMENTED;
        }
    } else {
        std::cerr << "  [ERROR] Unsupported SoC: " << SocVersionToString(socVersion)
                  << ", expected Ascend950" << std::endl;
        err = ACLFFT_EXEC_FAILED;
    }

    return (err == ACL_SUCCESS) ? ACLFFT_SUCCESS : ACLFFT_EXEC_FAILED;
}
} // extern "C"