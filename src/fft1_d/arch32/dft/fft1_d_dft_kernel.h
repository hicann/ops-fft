/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fft1_d.h
 * \brief
 */

#include "fft_common_kernel.h"
#include "fft1_d_dft_tilingdata.h"

using T_INPUT = float;
using T_OUTPUT = float;

extern "C" __global__ __aicore__ void dft(__gm__ T_INPUT *__restrict__ gm_a, __gm__ T_INPUT *__restrict__ gm_b,
                                          __gm__ T_OUTPUT *__restrict__ gm_c, __gm__ uint8_t *__restrict__ workspace,
                                          Fft1DDFTTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    SetPadding(0);
    AscendC::SetAtomicNone();
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    AscendC::GlobalTensor<T_INPUT> gm_a_tensor;
    AscendC::GlobalTensor<T_INPUT> gm_b_tensor;
    AscendC::GlobalTensor<T_INPUT> gm_c_tensor;
    AscendC::GlobalTensor<T_OUTPUT> workspace_tensor;

    gm_a_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_INPUT *>(gm_a));
    gm_b_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_INPUT *>(gm_b));
    gm_c_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_OUTPUT *>(gm_c));
    workspace_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T_OUTPUT *>(workspace));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<T_INPUT> l1_base_a = buf.GetBuffer<BufferType::ASCEND_CB, T_INPUT>(0);
    AscendC::LocalTensor<T_INPUT> l1_base_b= buf.GetBuffer<BufferType::ASCEND_CB, T_INPUT>(128 * 1024);

    AscendC::LocalTensor<T_INPUT> l0a_base = buf.GetBuffer<BufferType::ASCEND_L0A, T_INPUT>(0);
    AscendC::LocalTensor<T_INPUT> l0b_base = buf.GetBuffer<BufferType::ASCEND_L0B, T_INPUT>(0);
    AscendC::LocalTensor<float> l0c_base = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    int32_t batchSize = tilingData.batchSize;
    int32_t M = tilingData.m;
    int32_t N = tilingData.n;
    int32_t K = tilingData.k;
    int32_t trans_a = tilingData.transA;
    int32_t trans_b = tilingData.transB;
    int32_t M0 = tilingData.m0;
    if (M0 == 0) {
        M0 = 128;
    }
    int32_t N0 = tilingData.n0;
    if (N0 == 0) {
        N0 = 128;
    }
    int32_t K0 = tilingData.k0;
    if (K0 == 0) {
        K0 = 128;
    }

    int32_t m_loop = (M + M0 - 1) / M0;
    if (m_loop == 0) {
        m_loop = 1;
    }
    int32_t n_loop = (N + N0 - 1) / N0;
    if (n_loop == 0) {
        n_loop = 1;
    }
    int32_t k_loop = (K + K0 - 1) / K0;
    int32_t loop = batchSize * m_loop * n_loop;

    int32_t ping_flag = 1;

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);

    SET_FLAG(FIX, M, EVENT_ID0);
    SET_FLAG(FIX, M, EVENT_ID1);

    int32_t ping_flag_fix = 1;

    for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {
        if (loop_idx % AscendC::GetBlockNum() != AscendC::GetBlockIdx()) {
            continue;
        }

        auto l0c_buf = ping_flag_fix ? l0c_base : l0c_base[FFT_L0C_PINGPONG_LEN];
        auto event_id_fix = ping_flag_fix ? EVENT_ID0 : EVENT_ID1;

        int32_t batch_idx = loop_idx / (m_loop * n_loop);
        int32_t in_batch_idx = loop_idx % (m_loop * n_loop);

        constexpr int32_t N_COL = 16;
        int32_t tile_block_loop = (n_loop + N_COL - 1) / N_COL;
        int32_t tile_block_idx = in_batch_idx / (N_COL * m_loop);
        int32_t in_tile_block_idx = in_batch_idx % (N_COL * m_loop);
        int32_t n_col = N_COL;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = n_loop - N_COL * tile_block_idx;
            if (n_col == 0) {
                n_col = 1;
            }
        }
        int32_t m_idx = in_tile_block_idx / n_col;
        int32_t n_idx = tile_block_idx * N_COL + in_tile_block_idx % n_col;

        int64_t offset_a, offset_b;
        int64_t offset_c = static_cast<int64_t>(batch_idx) * M * N +
                           static_cast<int64_t>(m_idx) * M0 * K + static_cast<int64_t>(n_idx) * N0;  // // 修改代码，将矩阵乘结果的dst stride修改为K，之前是N。
        int32_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;
        int32_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
        int32_t m_round = (m_actual + 15) / 16 * 16;
        int32_t n_round = (n_actual + 15) / 16 * 16;

        int32_t mn_max = m_round > n_round ? m_round : n_round;
        int32_t k_part_len = FFT_L0AB_PINGPONG_LEN / mn_max / 16 * 16;

        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);

        int32_t mte1_mad_ping_flag = 1;

        for (int32_t k_idx = 0; k_idx < k_loop; k_idx++) {
            if (trans_a) {
                offset_a = static_cast<int64_t>(batch_idx) * M * K + static_cast<int64_t>(k_idx) * K0 * M + static_cast<int64_t>(m_idx) * M0;
            } else {
                offset_a = static_cast<int64_t>(batch_idx) * M * K + static_cast<int64_t>(m_idx) * M0 * K + static_cast<int64_t>(k_idx) * K0;
            }

            if (trans_b) {
                offset_b = static_cast<int64_t>(batch_idx) * K * N + static_cast<int64_t>(n_idx) * N0 * K + static_cast<int64_t>(k_idx) * K0;
            } else {
                offset_b = static_cast<int64_t>(batch_idx) * K * N + static_cast<int64_t>(k_idx) * K0 * N + static_cast<int64_t>(n_idx) * N0;
            }

            int32_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;
            int32_t k_round = (k_actual + 15) / 16 * 16;
            int32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

            auto l1_buf_a = ping_flag ? l1_base_a : l1_base_a[FFT_L1_PINGPONG_LEN];
            auto l1_buf_b = ping_flag ? l1_base_b : l1_base_b[FFT_L1_PINGPONG_LEN];
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            // *** load matrix A to L1
            WAIT_FLAG(MTE1, MTE2, event_id);
            if (trans_a) {
                if (M == 1) {
                    AscendC::DataCopy(
                        l1_buf_a, gm_a_tensor[offset_a],
                        AscendC::DataCopyParams(
                            1,                                      // nBurst
                            (k_actual + FFT_C0_SIZE - 1) / FFT_C0_SIZE,     // lenBurst
                            0,                                      // srcGap
                            0                                       // dstGap
                        )
                    );
                } else {
                    load_matrix_zZ<T_INPUT>(l1_buf_a, gm_a_tensor[offset_a], K0, M0, k_actual, m_actual, M);
                }
            } else {
                if (m_actual == 1) {
                    AscendC::DataCopy(
                        l1_buf_a, gm_a_tensor[offset_a],
                        AscendC::DataCopyParams(
                            1,                                      // nBurst
                            (k_actual + FFT_C0_SIZE - 1) / FFT_C0_SIZE,     // lenBurst
                            0,                                      // srcGap
                            0                                       // dstGap
                        )
                    );
                } else {
                    load_matrix_zZ<T_INPUT>(l1_buf_a, gm_a_tensor[offset_a], M0, K0, m_actual, k_actual, K);
                }
            }
            SET_FLAG(MTE2, MTE1, event_id);

            // *** load matrix B to L1
            WAIT_FLAG(MTE1, MTE2, event_id + 2);
            if (trans_b) {
                load_matrix_zZ<T_INPUT>(l1_buf_b, gm_b_tensor[offset_b], M0, K0, n_actual, k_actual, K);
            } else {
                load_matrix_zZ<T_INPUT>(l1_buf_b, gm_b_tensor[offset_b], K0, N0, k_actual, n_actual, N);
            }

            SET_FLAG(MTE2, MTE1, event_id + 2);

            for (int32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                int32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                int32_t k0_actual = (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                auto l0a_buf = l0a_base[mte1_mad_ping_flag * FFT_L0AB_PINGPONG_LEN];
                auto l0b_buf = l0b_base[mte1_mad_ping_flag * FFT_L0AB_PINGPONG_LEN];

                // *** load matrix A from L1 to L0A
                if (k_part_idx == 0) {
                    WAIT_FLAG(MTE2, MTE1, event_id);
                }
                WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                if (M == 1 || m_actual == 1 && !trans_a) {
#if (__CCE_AICORE__ == 220)
                    AscendC::LoadData(
                        l0a_buf,
                        l1_buf_a[k_part_idx * k_part_len],
                        AscendC::LoadData2dParams(
                            0,                                                      // baseIdx
                            (k0_round + FFT_CUBE_MATRIX_SIZE - 1) / FFT_CUBE_MATRIX_SIZE,   // repeat
                            1,                                                      // srcStride
                            0,                                                      // sid
                            0,                                                      // dstStride
                            false,                                                  // transpose
                            0                                                       // addr_cal_mode_t
                        )
                    );

#elif (__CCE_AICORE__ == 300)
                    AscendC::LoadData(
                        l0a_buf,
                        l1_buf_a[k_part_idx * k_part_len],
                        AscendC::LoadData2dParams(
                            0,                                                      // baseIdx
                            (k0_round + FFT_CUBE_MATRIX_SIZE - 1) / FFT_CUBE_MATRIX_SIZE,   // repeat
                            1,                                                      // srcStride
                            0,                                                      // sid
                            0,                                                      // dstStride
                            false,                                                  // transpose
                            0                                                       // addr_cal_mode_t
                        )
                    );

#endif
                } else {
                    if (trans_a) {
                        auto l1_src_a = l1_buf_a[k_part_idx * k_part_len * M0];
                        for (int32_t i = 0; i < m_round / FFT_BLOCK_SIZE; i++) {
                            AscendC::LoadDataWithTranspose(
                                l0a_buf[i * k0_round * FFT_BLOCK_SIZE],
                                l1_src_a[i * 2 * FFT_CUBE_MATRIX_SIZE],
                                AscendC::LoadData2dTransposeParams(
                                    0,                             // indexID
                                    k0_round / FFT_BLOCK_SIZE,         // repeat
                                    M0 / FFT_BLOCK_SIZE,               // srcStride
                                    1,                              // dstGap
                                    0,                              // dstFracStride
                                    inc                           // addrmode
                                )
                            );
                        }
                    } else {
                        auto l1_src_a = l1_buf_a[k_part_idx * k_part_len * FFT_BLOCK_SIZE];
                        for (int32_t i = 0; i < m_round / FFT_BLOCK_SIZE; i++) {
#if (__CCE_AICORE__ == 220)
                            AscendC::LoadData(
                                l0a_buf[i * k0_round * FFT_BLOCK_SIZE],
                                l1_src_a[i * FFT_BLOCK_SIZE * K0],
                                AscendC::LoadData2dParams(
                                    0,                   // baseIdx
                                    k0_round / FFT_C0_SIZE,  // repeat
                                    1,                   // srcStride
                                    0,                   // sid
                                    0,                   // dstGap
                                    false,               // transpose
                                    inc                  // addr_cal_mode_t
                                )
                            );

#elif (__CCE_AICORE__ == 300)
                            AscendC::LoadData(
                                l0a_buf[i * k0_round * FFT_BLOCK_SIZE],
                                l1_src_a[i * FFT_BLOCK_SIZE * K0],
                                AscendC::LoadData2dParams(
                                    0,                   // baseIdx
                                    k0_round / FFT_C0_SIZE,  // repeat
                                    1,                   // srcStride
                                    0,                   // sid
                                    0,                   // dstGap
                                    false,               // transpose
                                    inc                  // addr_cal_mode_t
                                )
                            );
#endif
                        }
                    }
                }
                if (k_part_idx == k_part_loop - 1) {
                    SET_FLAG(MTE1, MTE2, event_id);
                }

                // *** load matrix B from L1 to L0B
                if (k_part_idx == 0) {
                    WAIT_FLAG(MTE2, MTE1, event_id + 2);
                }
                if (trans_b) {
                    auto l1_src_b = l1_buf_b[k_part_idx * k_part_len * FFT_BLOCK_SIZE];
                    for (int32_t i = 0; i < k0_round / FFT_C0_SIZE; i++) {
#if (__CCE_AICORE__ == 220)
                        AscendC::LoadData(
                            l0b_buf[i * n_round * FFT_C0_SIZE],
                            l1_src_b[i * FFT_CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dParams(
                                0,                   // baseIdx
                                n_round / FFT_BLOCK_SIZE,  // repeat
                                K0 / FFT_C0_SIZE,                   // srcStride
                                0,                   // sid
                                0,                   // dstGap
                                false,               // transpose
                                inc                  // addr_cal_mode_t
                            )
                        );
#elif (__CCE_AICORE__ == 300)
                        AscendC::LoadData(
                            l0b_buf[i * n_round * FFT_C0_SIZE],
                            l1_src_b[i * FFT_CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dParams(
                                0,                   // baseIdx
                                n_round / FFT_BLOCK_SIZE,  // repeat
                                K0 / FFT_C0_SIZE,                   // srcStride
                                0,                   // sid
                                0,                   // dstGap
                                false,               // transpose
                                inc                  // addr_cal_mode_t
                            )
                        );
#endif
                    }
                } else {
                    auto l1_src_b = l1_buf_b[k_part_idx * k_part_len * N0];
                    for (int32_t i = 0; i < n_round / FFT_BLOCK_SIZE; i++) {
                        AscendC::LoadDataWithTranspose(
                            l0b_buf[i * FFT_CUBE_MATRIX_SIZE],
                            l1_src_b[i * 2 * FFT_CUBE_MATRIX_SIZE],
                            AscendC::LoadData2dTransposeParams(
                                0,                             // indexID
                                k0_round / FFT_BLOCK_SIZE,         // repeat
                                N0 / FFT_BLOCK_SIZE,               // srcStride
                                2 * n_round / FFT_BLOCK_SIZE - 1,  // dstGap
                                n_round / FFT_BLOCK_SIZE - 1,      // dstFracStride
                                inc                           // addrmode
                            )
                        );
                    }
                }
                if (k_part_idx == k_part_loop - 1) {
                    SET_FLAG(MTE1, MTE2, event_id + 2);
                }

                SET_FLAG(MTE1, M, mte1_mad_event_id);
                WAIT_FLAG(MTE1, M, mte1_mad_event_id);

                bool init_c = (k_idx == 0 && k_part_idx == 0);
                if (init_c) {
                    WAIT_FLAG(FIX, M, event_id_fix);
                }

                if (M != 1 && m_actual == 1 && trans_a) {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(
                        16, n_actual, k0_actual,
                        0,
                        false,
                        init_c
                    );
                    mmParam.kDirectionAlign = true;

                    AscendC::Mmad(
                        l0c_buf, l0a_buf, l0b_buf, mmParam
                    );
                } else {
                    AscendC::MmadParams mmParam = AscendC::MmadParams(
                        m_actual, n_actual, k0_actual,
                        0,
                        false,
                        init_c
                    );
                    mmParam.kDirectionAlign = true;

                    AscendC::Mmad(
                        l0c_buf, l0a_buf, l0b_buf, mmParam
                    );
                }

                PIPE_BARRIER(M);
                SET_FLAG(M, MTE1, mte1_mad_event_id);
                mte1_mad_ping_flag = 1 - mte1_mad_ping_flag;
            }

            ping_flag = 1 - ping_flag;
        }

        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);

        SET_FLAG(M, FIX, EVENT_ID0);
        WAIT_FLAG(M, FIX, EVENT_ID0);

        // copy from L0C to gm
        // 修改代码，将矩阵乘结果的dst stride修改为K，之前是N。修改为K的原因是因为右矩阵是一个方阵，
        // 完整的DFT的结果就是BachSize * K(这个K表示的是矩阵乘的K参数，这个K参数等于输入参数n。).
#if (__CCE_AICORE__ == 220)
        auto intriParams = AscendC::FixpipeParamsV220(n_actual, // nSize
                                                      m_actual, // mSize
                                                      m_round,   // srcStride
                                                      K,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;

        AscendC::Fixpipe(gm_c_tensor[offset_c], l0c_buf, intriParams);

#elif (__CCE_AICORE__ == 300)
        auto intriParams = AscendC::FixpipeParamsM300(n_actual, // nSize
                                                      m_actual, // mSize
                                                      m_round,   // srcStride
                                                      K,   // dstStride
                                                      false);      // enRelu

        intriParams.quantPre = QuantMode_t::NoQuant;

        AscendC::Fixpipe(gm_c_tensor[offset_c], l0c_buf, intriParams);
#endif
        SET_FLAG(FIX, M, event_id_fix);
        ping_flag_fix = 1 - ping_flag_fix;
    }

    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);

    WAIT_FLAG(FIX, M, EVENT_ID0);
    WAIT_FLAG(FIX, M, EVENT_ID1);

    PIPE_BARRIER(ALL);
}
