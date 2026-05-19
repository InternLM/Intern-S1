# Intern-S2-Preview 使用指南

## 采样超参

我们推荐使用如下的超参数以获得更好的生成效果：

```python
top_p = 0.95
top_k = 50
min_p = 0.0
temperature = 0.8
```

## 部署服务

Intern-S2-Preview 是一个 35B-A3B 模型，权重采用 bfloat16 格式存储。本指南提供以下几类部署配置示例：

- MTP 投机解码（推荐）
- 不启用 MTP 的基础服务
- 结合 YaRN RoPE 配置的长上下文推理

> 注：本指南中的部署示例仅供参考，并非最新或最优配置方案。推理框架仍在持续开发迭代中，请结合各框架维护方发布的官方文档和本地验证结果调整生产部署配置。

### LMDeploy

请使用支持 Intern-S2-Preview 的最新版 LMDeploy，推荐 `lmdeploy>=0.13.0`。

- 启用 MTP 的服务（推荐）

```bash
lmdeploy serve api_server \
    internlm/Intern-S2-Preview \
    --trust-remote-code \
    --backend pytorch \
    --tp 2 \
    --reasoning-parser default \
    --tool-call-parser interns2-preview \
    --speculative-algorithm qwen3_5_mtp \
    --speculative-num-draft-tokens 4 \
    --max-batch-size 256
```

- 不启用 MTP 的基础服务

```bash
lmdeploy serve api_server \
    internlm/Intern-S2-Preview \
    --trust-remote-code \
    --backend pytorch \
    --tp 2 \
    --reasoning-parser default \
    --tool-call-parser interns2-preview
```

- 长上下文服务

进行长上下文推理时，需要同时配置 `--session-len` 和 YaRN RoPE 参数。以下示例使用 512k 上下文长度：

```bash
lmdeploy serve api_server \
    internlm/Intern-S2-Preview \
    --trust-remote-code \
    --tp 2 \
    --backend pytorch \
    --reasoning-parser default \
    --tool-call-parser interns2-preview \
    --session-len 512000 \
    --max-batch-size 64 \
    --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

### vLLM

请使用支持 Intern-S2-Preview 的最新版 vLLM Docker 镜像或源码构建版本。

- 启用 MTP 的服务（推荐）

```bash
vllm serve internlm/Intern-S2-Preview \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

- 不启用 MTP 的基础服务

```bash
vllm serve internlm/Intern-S2-Preview \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

### SGLang

请使用支持 Intern-S2-Preview 的最新版 SGLang Docker 镜像或源码构建版本。

- 启用 MTP 的服务（推荐）

```bash
SGLANG_ENABLE_SPEC_V2=1 \
python3 -m sglang.launch_server \
  --model-path internlm/Intern-S2-Preview \
  --trust-remote-code \
  --tp-size 2 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-scheduler-strategy extra_buffer \
  --speculative-algo 'NEXTN' \
  --speculative-eagle-topk 1 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4
```

- 不启用 MTP 的基础服务

```bash
python3 -m sglang.launch_server \
    --model-path internlm/Intern-S2-Preview \
    --trust-remote-code \
    --tp-size 2 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

## Agent Framework 接入

Intern-S2-Preview 可以通过两种方式接入 agent framework：

- 连接自部署服务
- 调用官方 Intern API

下面分别给出 OpenAI-compatible agent framework（如 OpenClaw、Hermes 等）和 Claude Code 的接入示例。

### 自部署服务

首先使用 LMDeploy 启动模型服务。下面的示例假设服务运行在 `http://0.0.0.0:23333`。

如果需要工具调用能力，启动 LMDeploy 时请设置 `--tool-call-parser interns2-preview`，以确保工具调用能够被正确解析。

#### 接入 Agent Framework

大多数 agent framework 都支持 OpenAI-compatible endpoint。你可以将 framework 指向 LMDeploy 服务的 base URL：

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://0.0.0.0:23333/v1
export OPENAI_MODEL=internlm/Intern-S2-Preview
```

可以使用以下请求验证连接：

```bash
curl http://0.0.0.0:23333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "internlm/Intern-S2-Preview",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.8,
    "top_p": 0.95
  }'
```

#### 接入 Claude Code

LMDeploy 提供 Anthropic-compatible `/v1/messages` endpoint，Claude Code 可以直接连接该接口。将以下配置添加到 `~/.claude/settings.json`：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:23333",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_MODEL": "internlm/Intern-S2-Preview",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "internlm/Intern-S2-Preview"
  }
}
```

完整的验证、模型路由和故障排查流程可参考 [LMDeploy Claude Code 接入文档](https://lmdeploy.readthedocs.io/en/latest/intergration/claude_code.html)。

### 官方 Intern API

如果不希望自部署 Intern-S2-Preview，也可以使用官方 Intern API。请在 [internlm.intern-ai.org.cn](https://internlm.intern-ai.org.cn/) 注册并创建 API token，例如 `sk-xxxxxxxx`。

#### 接入 Agent Framework

官方服务兼容 OpenAI API，因此 agent framework 可以直接使用官方 endpoint。将 base URL 设置为 `https://chat.intern-ai.org.cn/api/v1`，模型名设置为 `intern-s2-preview`。

```bash
export OPENAI_API_KEY=sk-xxxxxxxx
export OPENAI_BASE_URL=https://chat.intern-ai.org.cn/api/v1
export OPENAI_MODEL=intern-s2-preview
```

可以使用以下请求验证连接：

```bash
curl https://chat.intern-ai.org.cn/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxxxxxxx" \
  -d '{
    "model": "intern-s2-preview",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.8,
    "top_p": 0.95
  }'
```

关于当前 endpoint、可用模型名、限流策略和高级参数，请参考 [Intern API 文档](https://internlm.intern-ai.org.cn/api/document?lang=zh)。

#### 接入 Claude Code

Claude Code 可以通过 Intern 的 Anthropic-compatible gateway 调用官方 Intern API：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://chat.intern-ai.org.cn",
    "ANTHROPIC_AUTH_TOKEN": "your-api-token",
    "ANTHROPIC_MODEL": "intern-s2-preview",
    "ANTHROPIC_SMALL_FAST_MODEL": "intern-s2-preview"
  }
}
```

随后使用以下命令启动 Claude Code：

```bash
claude --model intern-s2-preview
```

详细接入步骤请参考 [Intern API Claude Code 接入文档](https://internlm.intern-ai.org.cn/api/document/CLAUDE_CODE_INTEGRATION_EN/)。
