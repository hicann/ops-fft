/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rfft1_d.h"
#include "fft_exec_helper.h"

extern "C" {
aclfftResult aclfftExecR2C_1D(aclfftHandle plan,
                           aclfftReal* idata,
                           aclfftComplex* odata) {
    aclfftHandle_t* impl = plan;
    ACLFFT_EXEC_1D_ENTRY_CHECKS(impl, idata, odata, "R2C arch35");

    const uint32_t n = impl->lengths[0];
    const uint32_t batch = impl->batch;
    // norm 值域已与 rfft1_d.h 文档对齐（0=BACKWARD），normMode(0) 直传即为 BACKWARD（issue #82）
    int32_t rfft_norm = impl->normMode;
    int isForward = 1;

    aclError err;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    auto socVersion = ascendcPlatform->GetSocVersion();

    static constexpr uint32_t K_N_FFT_1024 = 1024;
    static constexpr uint32_t K_N_FFT_4096 = 4096;

    if (socVersion == platform_ascendc::SocVersion::ASCEND950) {
        std::vector<int64_t> factors = orderedFactorize(n);
        std::vector<int64_t> uniques = deDuplicates(factors);
        int radix = ChooseRadix(impl->type, uniques);

        // 优先级1（最高）：fft_r2c_multi_core
        // 原始约束 getR2CCore：n > K_N_FFT_1024 && radix==mix → kFftR2CArch35
        if (n > K_N_FFT_1024 && radix == K_RADIX_MIX) {
            err = aclfftRfft1DFft(reinterpret_cast<float*>(idata),
                                  reinterpret_cast<float*>(odata),
                                  n, rfft_norm, batch, isForward, impl->stream);
        }
        // 优先级2（最低，兜底）：fast_dft — n <= K_N_FFT_4096
        else if (n <= K_N_FFT_4096) {
            err = aclfftRfft1D(reinterpret_cast<float*>(idata),
                               reinterpret_cast<float*>(odata),
                               n, rfft_norm, batch, impl->stream);
        }
        else {
            std::cerr << "[ops-fft] R2C arch35: n=" << n << " radix=" << radix << " not implemented" << std::endl;
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
