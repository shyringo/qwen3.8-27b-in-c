<div align="center">

# 用笔记本 CPU 运行 Qwen3.8-27B

**面向本地聊天和工作的原生 C 语言推理引擎。**<br>
不需要 GPU、CUDA、Python、PyTorch、权重转换或其他推理框架。

**参数量 27B · 单 CPU · 峰值内存 15.24 GiB · 最优实测 TPOT 0.568 s/token（1.76 token/s）**

[English](README.md) · [架构](docs/ARCHITECTURE.md) · [推理优化](docs/OPTIMIZATIONS.md) · [正确性](docs/CORRECTNESS.md)

</div>

## 快速开始

Ubuntu、Debian 或 Windows WSL2：

```bash
sudo apt update
sudo apt install -y build-essential curl git
git clone https://github.com/shyringo/qwen3.8-27b-in-c.git
cd qwen3.8-27b-in-c
./qwen38.sh
```

macOS 需要先安装命令行开发工具和 OpenMP：

```bash
xcode-select --install
brew install libomp
git clone https://github.com/shyringo/qwen3.8-27b-in-c.git
cd qwen3.8-27b-in-c
./qwen38.sh
```

启动脚本会自动编译推理引擎，需要时从 ModelScope 断点续传 17.1 GB 的
Q4_K_M 权重，然后进入多轮对话。输入 `/reset` 开始新对话，输入 `/exit`
退出。

只运行一次推理：

```bash
./qwen38.sh --prompt "科技的边界在哪里？" --max-tokens 256
```

其他常用方式：

```bash
./qwen38.sh --prompt-file request.txt
./qwen38.sh --no-thinking
./qwen38.sh --reasoning-effort low
./qwen38.sh --system "你是一名严谨的 C 语言代码审查工程师。"
```

上下文长度和 CPU 线程数无需重新编译即可修改：

```bash
QWEN38_CONTEXT=8192 ./qwen38.sh
QWEN38_THREADS=6 ./qwen38.sh
```

运行固定的 32-token 贪心推理基准：

```bash
./scripts/benchmark.sh
```

默认上下文长度为 4,096 token，每轮最多生成 1,024 token，最多使用 12 个
CPU 线程。权重默认保存在 `~/.cache/qwen3.8-27b-in-c/model`，也可以通过
`QWEN38_MODEL_DIR` 指定其他位置。

## 环境要求

- 64 位 POSIX 系统、C99 编译器、`make` 和 `curl`。Windows 请使用 WSL2。
- 至少 18 GB 可用磁盘空间，用于存放固定版本的 Q4_K_M GGUF 权重。
- 建议为推理进程留出约 20 GiB 可用内存。4,096-token 上下文的实测峰值
  RSS 为 15.24 GiB；额外余量可以保证操作系统正常响应。
- 推荐使用支持 AVX2 的 x86-64 CPU。其他 64 位 POSIX CPU 可以使用经过
  测试的通用标量构建。

把模型放在 Linux 或 macOS 的原生文件系统中，更有利于操作系统缓存这份
17 GB 的内存映射文件。WSL2 的默认 `~/.cache` 位于 Linux 文件系统中。

## 实测性能

以下是固定基准的最优墙钟实测结果，不是理论估算：

| 模式 | TTFT | TPOT | 生成速度 |
|---|---:|---:|---:|
| 普通贪心解码（默认） | 4.001 s | **0.568 s/token** | **1.76 token/s** |
| MTP 验证，深度 3 | 5.349 s | 0.806 s/token | 1.24 token/s |
| MTP 验证，深度 4 | 5.233 s | 0.928 s/token | 1.08 token/s |

实测环境：Intel Core i5-1340P 笔记本、宿主机 32 GB 内存、Windows 11 +
WSL2 Ubuntu 22.04.5（WSL2 限制 24 GiB）、GCC 11.4、12 个 OpenMP 线程、
4,096-token 上下文、WSL2 ext4 文件系统上的固定 Q4_K_M 权重、已进入内存
页缓存的模型、固定 16-token 输入和 32 个贪心输出 token。使用默认的精确
SiLU 路径。`/usr/bin/time -v` 测得峰值 RSS 为 15,980,928 KiB
（15.24 GiB），没有使用 swap。

TTFT 对模型页缓存、磁盘、供电模式、温度和其他程序的负载非常敏感。TPOT
按可见 token 的实际就绪时间计算，包含采样和全部模型计算。当前基准中，
MTP 的验证成本高于接受收益，因此仍是可选实验功能，不会默认开启。

## 原生推理引擎

本项目直接用 C 语言实现 Qwen3.8-27B 计算图，并原位读取单文件 GGUF；不是
llama.cpp 或其他推理引擎的套壳。

- 完整 64 层主模型：48 层 Gated DeltaNet 和 16 层因果全注意力。
- 循环卷积与 DeltaNet 状态、KV cache、RoPE、RMSNorm、稠密 SwiGLU、
  Qwen 分词器、采样、思考强度控制和多轮对话。
- Q4_K/Q5_K/Q6_K 压缩权重直接计算、Q8_K 激活量化、AVX2/VNNI 整数
  点积和 OpenMP 输出行并行。
- 分层 prompt 批处理、多 token 共享权重解码、跨投影复用激活量化结果、
  权重内存映射和跨轮次驻留状态。
- 默认使用精确 SiLU；可选的事务式贪心 MTP 与普通主模型输出逐 token 一致。

完整实现与代码来源见[推理优化](docs/OPTIMIZATIONS.md)，计算图、状态、批处理
和内存布局见[架构](docs/ARCHITECTURE.md)。

## 推理正确性

正确性测试使用同一份固定的 Q4_K_M 权重。原生分层 prefill 与原生逐 token
执行对固定 prompt 产生字节完全一致的 248,320 维 logits；另一个不重排
权重的独立评估器给出相同的 top-5 token 顺序。原生事务式 MTP 也会逐 token
复现普通原生贪心解码。

```bash
make strict
make portable
```

模型哈希、oracle SHA、logits 对比、token ID 和测试范围见
[正确性文档](docs/CORRECTNESS.md)。

## 模型、许可证与致谢

启动脚本优先从 ModelScope 下载
[Unsloth 提供的 Qwen3.8-27B Q4_K_M GGUF](https://www.modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF)，
并提供 Hugging Face 备用下载地址。[Qwen3.8-27B 官方模型](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B)、
[Qwen3.8 项目](https://github.com/QwenLM/Qwen3.8)和本仓库均采用 Apache
License 2.0。模型权重下载到用户缓存中，不包含在仓库里。

Byte-level BPE 的基础代码改编自
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)。完整归属和
第三方说明见 [NOTICE](NOTICE)。
