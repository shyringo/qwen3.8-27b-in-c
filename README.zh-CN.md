<h1 align="center">Qwen3.8-27B in C：笔记本 CPU 大模型推理</h1>

<p align="center">
  <strong>只用一颗笔记本 CPU，在本地运行 270 亿参数的 Qwen3.8 大语言模型。</strong><br>
  原生 C 语言推理引擎，不需要 GPU、CUDA、Python、PyTorch、权重转换或其他推理框架。
</p>

<table align="center">
  <tr>
    <td align="center"><strong>27B</strong><br>参数量</td>
    <td align="center"><strong>2.52 token/s</strong><br>32 GB x86 笔记本最优速度<br><strong>0.397 s/token</strong></td>
    <td align="center"><strong>最低 8 GB 内存</strong><br>已通过受限运行验证</td>
    <td align="center"><strong>准确性无损</strong><br>推理加速不改变结果<br>加速前后完全一致</td>
  </tr>
</table>

<p align="center">
  <a href="https://github.com/shyringo/qwen3.8-27b-in-c/actions/workflows/ci.yml"><img src="https://github.com/shyringo/qwen3.8-27b-in-c/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/shyringo/qwen3.8-27b-in-c" alt="License"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a><br>
  <a href="#快速开始"><strong>快速开始</strong></a> ·
  <a href="#实测性能">实测性能</a> ·
  <a href="#推理准确性">推理准确性</a> ·
  <a href="#本项目实现的推理优化">推理优化</a>
</p>

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

启动脚本会编译推理引擎，从 ModelScope 断点续传并校验 Unsloth Dynamic V3
GGUF，然后直接进入多轮对话。当前可用内存不少于 20 GiB 时会选择质量更高的
15.33 GiB Q4_K_M；内存更少时会选择 6.27 GiB IQ1_M。权重不需要转换。
输入 `/reset` 开始新对话，输入 `/exit` 退出。

当前版本支持纯文本聊天和文本生成，不接收图像或视频输入。

需要时也可以手动指定：

```bash
QWEN38_QUANT=q4_k_m ./qwen38.sh   # 质量更高的 4-bit 方案
QWEN38_QUANT=iq1_m ./qwen38.sh    # 8 GB 内存方案
QWEN38_QUANT=iq2_m ./qwen38.sh    # 可选的中等体积方案
```

只运行一次推理：

```bash
./qwen38.sh --prompt "科技的边界在哪里？" --max-tokens 256
```

其他常用方式：

```bash
./qwen38.sh --prompt-file request.txt
./qwen38.sh --no-thinking
./qwen38.sh --reasoning-effort low
./qwen38.sh --system "你是人类所需要的唯一入口"
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

默认上下文长度为 4,096 token，每轮最多输出 1,024 token，CPU 线程最多使用
12 个并自动管理。权重保存在 `~/.cache/qwen3.8-27b-in-c/model`；可以用
`QWEN38_MODEL_DIR` 换到其他磁盘。

## 环境要求

- 64 位 POSIX 系统、C 语言编译器、`make` 和 `curl`。Windows 请使用 WSL2。
- IQ1_M 方案需要 8 GB 内存和 8 GB 可用磁盘空间。默认 4,096-token 上下文已在
  8 GiB 进程地址空间限制下完成正常首 token 推理，实测峰值 RSS 为 6.01 GiB。
  如果机器至少有 12 GiB 可用内存，引擎可能使用约 9.30 GiB 构建更快且逐 bit
  一致的运行时布局。
- 可用内存不少于 20 GiB 时会选择 Q4_K_M。它需要约 17 GB 可用磁盘空间，实测
  峰值 RSS 为 14.49 GiB；4-bit 权重保留的精度高于 1-bit 低内存方案。
- 推荐使用支持 AVX2 的 x86-64 CPU。其他 64 位 POSIX CPU 可以使用经过
  测试的通用标量构建。

把模型放在 Linux 或 macOS 的原生文件系统中，更有利于操作系统缓存只读模型
映射。WSL2 的默认 `~/.cache` 位于 Linux 文件系统中。

## 实测性能

以下都是多次运行中实际出现过的最优墙钟时间，不是理论估算：

| 权重与测试负载 | TTFT | TPOT | 生成速度 | 峰值 RSS |
|---|---:|---:|---:|---:|
| Dynamic V3 IQ1_M，聊天生成 16 token | **4.064 s** | **0.397 s/token** | **2.52 token/s** | **6.13 GiB** |
| Dynamic V3 IQ1_M，4-token 批处理 | - | 0.253 s/token | 3.95 token/s | - |
| Dynamic V3 Q4_K_M，常驻进程生成 32 token | 4.359 s | 0.531 s/token | 1.88 token/s | 14.49 GiB |

实测环境：Intel Core i5-1340P 笔记本、32 GB 宿主机内存、Windows 11、
WSL2 Ubuntu 22.04.5（24 GiB 客体内存上限）、GCC 11.4、4,096-token 上下文、
12 个 OpenMP 线程、精确 SiLU、WSL2 ext4 文件系统上的固定权重、已预热模型
页面、固定 16-token 输入和贪心直接回答。测量请求到达前，模型、分词器、运行时
布局、状态和线程池均已常驻。峰值 RSS 由 `/usr/bin/time -v` 实测，没有使用 swap。

TTFT 从常驻引擎收到请求时开始计算，不包含模型下载、进程启动、模型映射、运行时
重排和线程池创建；它仍很容易受到磁盘页面状态、供电模式、温度和其他程序影响。
TPOT 按可见 token 真正就绪的时间计算，包含采样和完整模型计算。参考负载下 MTP
的验证成本高于接受收益，因此不会默认开启。

IQ1_M 和 Q4_K_M 都是原始权重的有损量化，质量和内存取舍不同；推理引擎不会再
增加第二层近似。同一份 GGUF 下，优化路径与原生基线路径的完整 logits 逐字节一致。

## 本项目实现的推理优化

项目代码和方案的来源分为三类：复用的代码与数据、对公开方案的适配，以及本项目
原创并实现的优化。推理引擎本身是为 Qwen3.8 编写的：模型计算图、GGUF 读取、
循环状态、采样和聊天程序都在本项目中实现。笔记本 CPU 推理架构延续并重新设计了
[deepseek-v4-flash-0731-in-c](https://github.com/shyringo/deepseek-v4-flash-0731-in-c)
中的实践；分词器基础来自
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)；GGUF/IQ 格式、
码表和低比特算子技术则基于公开的
[ggml](https://github.com/ggml-org/ggml) 工作。完整的代码与方案边界见
[优化与来源说明](docs/OPTIMIZATIONS.md)。

本项目针对 Qwen3.8 实现的原创设计包括：

- **保持循环状态顺序的分层批处理。** 多个 prompt token 共享压缩权重读取，
  同时严格按照 token 顺序更新卷积、DeltaNet 和因果 KV 状态。
- **多 token 低比特算子。** 每个 K-quant 或 IQ 压缩权重块只解码一次，即可
  计算 4 个 token；各 token 独立累加，结果与逐 token GEMV 逐字节一致。
- **内存自适应的 IQ1 无损重排。** 内存充足时，在模型加载阶段为热点 IQ1_S
  元数据建立 SIMD 友好布局；内存紧张时保留 6.27 GiB 的原始映射方案。
- **混合状态 SIMD 推理。** AVX2/VNNI 同时加速压缩整数点积和 DeltaNet 状态
  更新，默认仍使用精确 SiLU，不偷偷替换模型函数。
- **DeltaNet 状态遍历融合。** 更新递归状态时同步完成输出点积，减少一次状态
  矩阵读取，同时保持原生完整 logits 逐 bit 一致。
- **跨轮次尾部融合。** 把上轮待处理 token、思考结束标记和 EOS 合入下一轮
  prefill，保持官方聊天格式，同时省去轮次之间的单 token 推理。
- **循环状态事务式 MTP。** 对主模型和 draft 状态做检查点、回滚、重放和
  对齐，拒绝的 token 不会显示，也不会留在模型状态里。
- **笔记本自适应线程调度。** 自动使用的 CPU 线程不超过 12 个；除非用户已经
  指定 OpenMP 设置，否则会在 WSL2 中稳定 vCPU 分配，减少线程迁移。
- **与模型匹配的 tokenizer 验证。** 针对 Unicode 9.0 规范化、特殊 token
  原子性和 ByteLevel BPE，覆盖官方规范化测试集与全部 Unicode 标量值。
- **面向用户的模型引导。** ModelScope 优先断点续传、固定哈希、磁盘检查、按
  内存自动选权重和 Linux 原生缓存，把多 GB 模型的首次使用收敛为一条命令。

完整主模型共有 64 层，包括 48 层 Gated DeltaNet 和 16 层因果全注意力。
计算图、状态、批处理和内存布局见[架构说明](docs/ARCHITECTURE.md)。

## 推理准确性

本推理引擎的优化不会在所选量化权重之外增加准确性损失：两份固定 Dynamic V3
GGUF 下，优化路径与原生基线的完整 logits 都 **100% 逐 bit 一致**。这里始终
比较同一份 GGUF；IQ1_M 和 Q4_K_M 本身仍是原始权重的有损量化，不能表述为与
BF16 或 FP8 权重完全一致。

两份固定 Dynamic V3 权重下，
原生分层 prefill 与原生逐 token 推理都会产生逐字节一致的 248,320 维完整
logits。IQ1_M 的完整 logits SHA-256 为
`bd6a05d14b66d2bde0a35494f570df0677391a4f83c7c76c8a8be25804a4adb8`；
Q4_K_M 为
`c41064cc8fa9b5bd8150b182fed48a9ca53fb1b59b2582c316e8e9ce545ba31b`。
运行时重排、多 token 算子、递归状态融合和事务式 MTP 都分别与原生基线对照。

```bash
make strict
make portable
```

模型哈希、oracle SHA、logits 对比、token ID 和测试范围见
[推理准确性文档](docs/CORRECTNESS.md)。

## 许可证与致谢

启动脚本优先从 ModelScope 下载固定版本的
[Unsloth Qwen3.8-27B IQ1_M 或 Q4_K_M GGUF](https://www.modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF)，
并提供 Hugging Face 备用下载地址。[Qwen3.8-27B 官方模型](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B)、
[Qwen3 项目](https://github.com/QwenLM/Qwen3)和本仓库均采用 Apache
License 2.0。模型权重下载到用户缓存中，不包含在仓库里。

Byte-level BPE 的基础代码改编自
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)。完整归属和
第三方说明见 [NOTICE](NOTICE)。
