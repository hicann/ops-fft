#!/usr/bin/env python3
"""
生成 rfft 预期值的 Python 脚本（使用标准输入管道）
输入：从 stdin 读取二进制 float 数据
输出：向 stdout 写入二进制 float 数据
"""

import numpy as np
import sys
import struct


def compute_rfft(x, norm=1):
    """
    计算 RFFT 预期值
    
    Args:
        x: 输入数组
        norm: 归一化选项 (1=backward, 2=forward, 3=ortho)
    
    Returns:
        预期值数组（实部和虚部交错）
    """
    x = np.array(x, dtype=np.float64)
    result = np.fft.rfft(x)
    
    # 应用归一化
    n = len(x)
    if norm == 2:  # forward
        result = result / n
    elif norm == 3:  # ortho
        result = result / np.sqrt(n)
    
    # 转换为实部和虚部交错
    expected = []
    for val in result:
        expected.append(float(val.real))
        expected.append(float(val.imag))
    
    return expected


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 generate_rfft_expected_pipe.py <n> <batches> <norm>", file=sys.stderr)
        print("  n: input length per batch", file=sys.stderr)
        print("  batches: number of batches", file=sys.stderr)
        print("  norm: normalization mode (1=backward, 2=forward, 3=ortho)", file=sys.stderr)
        print("  Input data is read from stdin as binary floats", file=sys.stderr)
        print("  Output is written to stdout as binary floats", file=sys.stderr)
        return 1
    
    n = int(sys.argv[1])
    batches = int(sys.argv[2])
    norm = int(sys.argv[3])
    
    # 从标准输入读取输入数据
    data = sys.stdin.buffer.read()
    num_floats = len(data) // 4
    x = list(struct.unpack(f'{num_floats}f', data))
    
    # 计算预期值
    expected = []
    for b in range(batches):
        batch_x = x[b * n : (b + 1) * n]
        batch_result = compute_rfft(batch_x, norm)
        expected.extend(batch_result)
    
    # 输出预期值到标准输出（二进制格式）
    sys.stdout.buffer.write(struct.pack(f'{len(expected)}f', *expected))
    sys.stdout.buffer.flush()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
