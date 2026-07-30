/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFT1_D_MIX_KERNEL_H
#define FFT1_D_MIX_KERNEL_H

#include "fft_c2c_common.h"

namespace FFT1DMixKernel {

extern "C" __global__ __aicore__ void fft_mix_1(
    __gm__ uint8_t *__restrict__ ffts_addr, __gm__ float *__restrict__ gm_input,
    __gm__ float *__restrict__ gm_dft_matrix_array, __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list, __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace, __gm__ uint8_t *__restrict__ gm_tiling_para)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_mix_cube(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                        gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#elif __DAV_C220_VEC__
{
    common_fft_mix_vector<false, 1>(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                    gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#endif
}

extern "C" __global__ __aicore__ void fft_mix_2(
    __gm__ uint8_t *__restrict__ ffts_addr, __gm__ float *__restrict__ gm_input,
    __gm__ float *__restrict__ gm_dft_matrix_array, __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list, __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace, __gm__ uint8_t *__restrict__ gm_tiling_para)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_mix_cube(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                        gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#elif __DAV_C220_VEC__
{
    common_fft_mix_vector<false, 2>(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                    gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#endif
}

extern "C" __global__ __aicore__ void fft_mix_3(
    __gm__ uint8_t *__restrict__ ffts_addr, __gm__ float *__restrict__ gm_input,
    __gm__ float *__restrict__ gm_dft_matrix_array, __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list, __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace, __gm__ uint8_t *__restrict__ gm_tiling_para)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_mix_cube(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                        gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#elif __DAV_C220_VEC__
{
    common_fft_mix_vector<false, 3>(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                    gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#endif
}

extern "C" __global__ __aicore__ void fft_mix_4(
    __gm__ uint8_t *__restrict__ ffts_addr, __gm__ float *__restrict__ gm_input,
    __gm__ float *__restrict__ gm_dft_matrix_array, __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list, __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace, __gm__ uint8_t *__restrict__ gm_tiling_para)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_mix_cube(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                        gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#elif __DAV_C220_VEC__
{
    common_fft_mix_vector<true, 1>(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                   gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#endif
}

extern "C" __global__ __aicore__ void fft_mix_5(
    __gm__ uint8_t *__restrict__ ffts_addr, __gm__ float *__restrict__ gm_input,
    __gm__ float *__restrict__ gm_dft_matrix_array, __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list, __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace, __gm__ uint8_t *__restrict__ gm_tiling_para)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_mix_cube(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                        gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#elif __DAV_C220_VEC__
{
    common_fft_mix_vector<true, 2>(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                   gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#endif
}

extern "C" __global__ __aicore__ void fft_mix_6(
    __gm__ uint8_t *__restrict__ ffts_addr, __gm__ float *__restrict__ gm_input,
    __gm__ float *__restrict__ gm_dft_matrix_array, __gm__ float *__restrict__ gm_tw_matrix_array,
    __gm__ int32_t *__restrict__ gm_radix_list, __gm__ float *__restrict__ gm_output,
    __gm__ float *__restrict__ gm_workspace, __gm__ uint8_t *__restrict__ gm_tiling_para)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#ifdef __DAV_C220_CUBE__
{
    common_fft_mix_cube(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                        gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#elif __DAV_C220_VEC__
{
    common_fft_mix_vector<true, 3>(ffts_addr, gm_input, gm_dft_matrix_array, gm_tw_matrix_array,
                                   gm_radix_list, gm_output, gm_workspace, gm_tiling_para);
}
#endif
}

} // namespace FFT1DMixKernel

#endif // FFT1_D_MIX_KERNEL_H
