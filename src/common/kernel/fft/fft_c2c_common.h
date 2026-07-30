/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fft_all_common.h"

template <bool is_vec_vtranspose_load, int32_t aiv_split_way>
__aicore__ __inline__ void
common_fft_mix_vector(__gm__ uint8_t *__restrict__ ffts_addr, __gm__ float *__restrict__ gm_input,
                      __gm__ float *__restrict__ gm_dft_matrix_array, __gm__ float *__restrict__ gm_tw_matrix_array,
                      __gm__ int32_t *__restrict__ gm_radix_list, __gm__ float *__restrict__ gm_output,
                      __gm__ float *__restrict__ gm_workspace, __gm__ uint8_t *__restrict__ gm_tiling_para)
{
    Init4Vector(ffts_addr);

    OpsFft::FftAllMixTilingData tiling_data;
    InitTilingData(gm_tiling_para, &tiling_data);

    __gm__ float *__restrict__ gm_workspace_input =
        (__gm__ float *__restrict__)((__gm__ uint8_t *__restrict__)gm_workspace + tiling_data.workspaceOffsets[0]);
    __gm__ float *__restrict__ gm_workspace_output =
        (__gm__ float *__restrict__)((__gm__ uint8_t *__restrict__)gm_workspace + tiling_data.workspaceOffsets[1]);
    __gm__ float *__restrict__ gm_workspace_sync =
        (__gm__ float *__restrict__)((__gm__ uint8_t *__restrict__)gm_workspace + tiling_data.workspaceOffsets[2]);
    __gm__ float *__restrict__ gm_c2c_output =
        (__gm__ float *__restrict__)((__gm__ uint8_t *__restrict__)gm_workspace + tiling_data.workspaceOffsets[3]);
    __gm__ float *__restrict__ gm_auxil =
        (__gm__ float *__restrict__)((__gm__ uint8_t *__restrict__)gm_workspace + tiling_data.workspaceOffsets[4]);

    stock_fft_mix_aiv<is_vec_vtranspose_load, aiv_split_way>(
        gm_input, gm_dft_matrix_array, gm_tw_matrix_array, gm_workspace_input, gm_workspace_output, gm_workspace_sync,
        gm_radix_list, gm_output, gm_auxil, tiling_data.batchSize, tiling_data.fftN, tiling_data.radixListLen,
        tiling_data.isInverse);
}
