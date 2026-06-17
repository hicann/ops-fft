#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'common'))
import test_helper

logger = logging.getLogger(__name__)

configs = [
    {"name": "c2c_forward_n512", "n": 512, "batch": 1, "forward": True},
    {"name": "c2c_forward_n420", "n": 420, "batch": 1, "forward": True},
    {"name": "c2c_roundtrip_n512", "n": 512, "batch": 1, "roundtrip": True},
    {"name": "dft_forward", "n": 4, "batch": 1, "forward": True},
    {"name": "dft_backward", "n": 4, "batch": 1, "forward": False},
    {"name": "dft_config", "n": 256, "batch": 10, "forward": True},
    {"name": "stride_forward", "n": 256, "batch": 128, "forward": True, "vertical": True},
    {"name": "fft_n_forward", "n": 32768, "batch": 1, "forward": True},
    {"name": "fft_n_roundtrip", "n": 32768, "batch": 1, "roundtrip": True},
    {"name": "fft_n_config", "n": 32768, "batch": 10, "forward": True},
    {"name": "fft_b_n512", "n": 512, "batch": 2, "forward": True},
    {"name": "fft_b_n2048", "n": 2048, "batch": 2, "forward": True},
    {"name": "fft_b_n16384", "n": 16384, "batch": 2, "forward": True},
]


def gen_one(cfg, seed):
    name = cfg["name"]
    n = cfg["n"]
    batch = cfg["batch"]
    total = n * batch
    torch.manual_seed(seed)
    input_complex = torch.randn(batch, n, dtype=torch.complex64)
    input_flat = np.empty(total * 2, dtype=np.float32)
    input_flat[0::2] = input_complex.reshape(-1).real.numpy()
    input_flat[1::2] = input_complex.reshape(-1).imag.numpy()
    is_vertical = cfg.get("vertical", False)
    if cfg.get("roundtrip"):
        golden_forward = torch.fft.fft(input_complex, dim=1)
        golden_flat_forward = np.empty(total * 2, dtype=np.float32)
        golden_flat_forward[0::2] = golden_forward.reshape(-1).real.numpy()
        golden_flat_forward[1::2] = golden_forward.reshape(-1).imag.numpy()
        roundtrip_result = torch.fft.ifft(golden_forward, dim=1)
        golden_flat = np.empty(total * 2, dtype=np.float32)
        golden_flat[0::2] = roundtrip_result.reshape(-1).real.numpy() * n
        golden_flat[1::2] = roundtrip_result.reshape(-1).imag.numpy() * n
    elif is_vertical:
        input_vertical = input_complex.reshape(n, batch)
        golden = torch.fft.fft(input_vertical, dim=0) if cfg["forward"] else torch.fft.ifft(input_vertical, dim=0) * n
        golden_flat = np.empty(total * 2, dtype=np.float32)
        golden_flat[0::2] = golden.reshape(-1).real.numpy()
        golden_flat[1::2] = golden.reshape(-1).imag.numpy()
    else:
        golden = torch.fft.fft(input_complex, dim=1) if cfg["forward"] else torch.fft.ifft(input_complex, dim=1) * n
        golden_flat = np.empty(total * 2, dtype=np.float32)
        golden_flat[0::2] = golden.reshape(-1).real.numpy()
        golden_flat[1::2] = golden.reshape(-1).imag.numpy()
    input_flat.tofile(f"{name}_input.bin")
    golden_flat.tofile(f"{name}_golden.bin")
    logger.info(f"  Generated: {name}_input.bin ({len(input_flat)} floats), "
                f"{name}_golden.bin ({len(golden_flat)} floats)")
    if cfg.get("roundtrip"):
        golden_flat_forward.tofile(f"{name}_golden_forward.bin")
        logger.info(f"  Generated: {name}_golden_forward.bin ({len(golden_flat_forward)} floats)")

if __name__ == "__main__":
    sys.exit(test_helper.gen_main(configs, gen_one))
