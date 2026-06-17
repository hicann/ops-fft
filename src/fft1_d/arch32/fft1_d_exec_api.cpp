/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fft1_d.h"
#include "fft_exec_helper.h"

enum class FftCoreSelect : int { kNone = 0, kDft = 1, kFftB = 2, kFftN = 3 };
struct FftConfigEntry {
    uint32_t batchLower;
    uint32_t batchUpper;
    std::vector<int64_t> dftRadices;
    std::vector<int64_t> fftBRadices;
    std::vector<int64_t> fftNRadices;
};

static std::vector<FftConfigEntry> &GetFftConfigs()
{
    static std::vector<FftConfigEntry> configs = {
        {8,    96,     {256, 512, 1024}, {2048, 4096, 8192, 16384}, {32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304}},
        {96,   1024,   {256, 512},       {1024, 2048, 4096, 8192, 16384}, {}},
        {1024, 65537,  {256},            {512, 1024, 2048, 4096, 8192, 16384}, {}}
    };
    return configs;
}

static FftCoreSelect FindConfig(uint32_t batch, int64_t n)
{
    for (const auto &entry : GetFftConfigs()) {
        if (batch >= entry.batchLower && batch < entry.batchUpper) {
            if (std::find(entry.dftRadices.begin(), entry.dftRadices.end(), n) != entry.dftRadices.end()) {
                return FftCoreSelect::kDft;
            }
            if (std::find(entry.fftBRadices.begin(), entry.fftBRadices.end(), n) != entry.fftBRadices.end()) {
                return FftCoreSelect::kFftB;
            }
            if (std::find(entry.fftNRadices.begin(), entry.fftNRadices.end(), n) != entry.fftNRadices.end()) {
                return FftCoreSelect::kFftN;
            }
            return FftCoreSelect::kNone;
        }
    }
    return FftCoreSelect::kNone;
}

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

    if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        if (impl->stride[0] > 1) {
            err = aclfftFft1DStride(reinterpret_cast<float*>(idata),
                                    reinterpret_cast<float*>(odata),
                                    n, impl->stride[0], batch, isForward, impl->stream);
        } else {
            int radix = ChooseRadix(impl->type, uniques);
            FftCoreSelect configCore = FindConfig(batch, static_cast<int64_t>(n));
            if (configCore == FftCoreSelect::kDft) {
                err = aclfftFft1DDft(reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata),
                                     n, fft_norm, batch, isForward, impl->stream);
            } else if (configCore == FftCoreSelect::kFftB) {
                err = aclfftFft1DB(reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata),
                                   n, batch, isForward, impl->stream);
            } else if (configCore == FftCoreSelect::kFftN) {
                err = aclfftFft1DN(reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata),
                                   n, fft_norm, batch, isForward, impl->stream);
            } else if (n <= K_N_FFT_256) {
                err = aclfftFft1DDft(reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata),
                                     n, fft_norm, batch, isForward, impl->stream);
            } else if (radix == K_RADIX_2) {
                if (n < K_N_FFT_32768) {
                    err = aclfftFft1DB(reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata),
                                       n, batch, isForward, impl->stream);
                } else {
                    err = aclfftFft1DN(reinterpret_cast<float*>(idata), reinterpret_cast<float*>(odata),
                                       n, fft_norm, batch, isForward, impl->stream);
                }
            } else {
                std::cerr << "[ops-fft] C2C arch32: n=" << n << " C2C exec path not implemented" << std::endl;
                return ACLFFT_NOT_IMPLEMENTED;
            }
        }
    } else {
        std::cerr << "  [ERROR] Unsupported SoC: " << SocVersionToString(socVersion)
                  << ", expected Ascend910B" << std::endl;
        err = ACLFFT_EXEC_FAILED;
    }

    return (err == ACL_SUCCESS) ? ACLFFT_SUCCESS : ACLFFT_EXEC_FAILED;
}
} // extern "C"
