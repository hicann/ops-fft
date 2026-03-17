/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of conditions of the CANN Open Software License Agreement Version 2.0.
 */

/**
 * @file rfft_expected_loader_pipe.h
 * @brief RFFT 预期值生成器（使用双向管道）
 */

#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cerrno>
#include <cstring>
#include <unistd.h>
#include <sys/wait.h>

namespace RfftExpected {

/**
 * @brief 调用 Python 脚本生成预期值（使用双向管道）
 * @param x 输入数据
 * @param n 每个 batch 的长度
 * @param batches batch 数量
 * @param norm 归一化选项
 * @param expected 输出预期值
 */
inline void generateExpectedData(const std::vector<float>& x, int64_t n, uint32_t batches, int64_t norm, std::vector<float>& expected) {
    // 创建两个管道：一个用于 stdin，一个用于 stdout
    int stdin_pipe[2], stdout_pipe[2];
    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
        throw std::runtime_error("Failed to create pipes");
    }
    
    // Fork 进程
    pid_t pid = fork();
    if (pid < 0) {
        close(stdin_pipe[0]); close(stdin_pipe[1]);
        close(stdout_pipe[0]); close(stdout_pipe[1]);
        throw std::runtime_error("Failed to fork process");
    }
    
    if (pid == 0) {
        // 子进程：Python 脚本
        close(stdin_pipe[1]);  // 关闭写入端
        close(stdout_pipe[0]); // 关闭读取端
        
        // 重定向 stdin 和 stdout
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        
        // 关闭原始文件描述符
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        
        // 执行 Python 脚本
        std::string python_script = "../src/rfft1_d/tests/scripts/generate_rfft_expected.py";
        std::string cmd = "python3 " + python_script + " " + 
                          std::to_string(n) + " " + 
                          std::to_string(batches) + " " + 
                          std::to_string(norm);
        
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char*)NULL);
        _exit(127);  // exec 失败
    }
    
    // 父进程：C++
    close(stdin_pipe[0]);  // 关闭读取端
    close(stdout_pipe[1]);  // 关闭写入端
    
    std::cout << "Generating expected data with Python (pipe mode)..." << std::endl;
    
    // 写入输入数据到 Python 脚本的标准输入
    size_t bytes_written = write(stdin_pipe[1], x.data(), x.size() * sizeof(float));
    if (bytes_written != x.size() * sizeof(float)) {
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        waitpid(pid, NULL, 0);
        throw std::runtime_error("Failed to write all data to Python script");
    }
    
    // 关闭 stdin 写入端，通知 Python 输入结束
    close(stdin_pipe[1]);
    
    // 计算预期输出大小
    int64_t expected_size = (n / 2 + 1) * 2 * batches;
    expected.resize(expected_size);
    
    // 从 Python 脚本的标准输出读取预期值
    size_t bytes_read = 0;
    size_t total_bytes = expected_size * sizeof(float);
    char* buffer = reinterpret_cast<char*>(expected.data());
    
    while (bytes_read < total_bytes) {
        ssize_t result = read(stdout_pipe[0], buffer + bytes_read, total_bytes - bytes_read);
        if (result < 0) {
            close(stdout_pipe[0]);
            waitpid(pid, NULL, 0);
            throw std::runtime_error("Failed to read from Python script");
        }
        if (result == 0) {
            break;  // EOF
        }
        bytes_read += result;
    }
    
    // 关闭 stdout 读取端
    close(stdout_pipe[0]);
    
    // 等待子进程结束
    int status;
    waitpid(pid, &status, 0);
    
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        throw std::runtime_error("Python script failed with return code: " + std::to_string(WEXITSTATUS(status)));
    }
    
    if (bytes_read != total_bytes) {
        throw std::runtime_error("Failed to read expected data from Python script: expected " + 
                                std::to_string(expected_size) + " floats, got " + 
                                std::to_string(bytes_read / sizeof(float)));
    }
    
    std::cout << "Expected data generated successfully." << std::endl;
}

} // namespace RfftExpected
