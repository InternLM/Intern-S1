# Intern-S2-Preview-397B User Guide

## Sampling Parameters

We recommend using the following hyperparameters to ensure better results.

```python
top_p = 0.95
top_k = 50
min_p = 0.0
temperature = 0.8
```

## Serving

We recommend deploying Intern-S2-Preview-397B on H100 (x8) or H200 (x8) nodes. This guide provides deployment examples for the following configurations:

- Basic serving without MTP
- MTP speculative decoding
- Long-context inference with YaRN RoPE configuration

> NOTE: The deployment examples in this guide are provided for reference only and may not represent the latest or most optimized configurations. Inference frameworks are under active development, so always consult the official documentation from each framework's maintainers and validate the configuration in your local environment.

### LMDeploy (>=0.14.0)

- Basic Serving Without MTP

```bash
# proxy server
lmdeploy serve proxy --server-name ${proxy_server_ip} --server-port ${proxy_server_port}

# api_server
lmdeploy serve api_server \
    internlm/Intern-S2-Preview-397B \
    --trust-remote-code \
    --backend pytorch \
    --dp 4 \
    --ep 8 \
    --enable-prefix-caching \
    --proxy-url http://${proxy_server_ip}:${proxy_server_port} \
    --reasoning-parser default \
    --tool-call-parser interns2-preview
```

- Serving With MTP

```bash
lmdeploy serve api_server \
    internlm/Intern-S2-Preview-397B \
    --trust-remote-code \
    --backend pytorch \
    --dp 4 \
    --ep 8 \
    --enable-prefix-caching \
    --proxy-url http://${proxy_server_ip}:${proxy_server_port} \
    --reasoning-parser default \
    --tool-call-parser interns2-preview \
    --speculative-algorithm qwen3_5_mtp \
    --speculative-num-draft-tokens 4 \
    --max-batch-size 256
```

- Long-Context Serving

For long-context inference, configure both `--session-len` and YaRN RoPE parameters. The following example uses a 512k context length:

```bash
lmdeploy serve api_server \
    internlm/Intern-S2-Preview-397B \
    --trust-remote-code \
    --backend pytorch \
    --dp 4 \
    --ep 8 \
    --enable-prefix-caching \
    --reasoning-parser default \
    --tool-call-parser interns2-preview \
    --session-len 512000 \
    --max-batch-size 64 \
    --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

### vLLM (>=v0.22.1)

- Basic Serving Without MTP

```bash
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_USE_DEEP_GEMM=0
export VLLM_FLASHINFER_MOE_BACKEND=latency

vllm serve internlm/Intern-S2-Preview-397B \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data
```

- Serving With MTP

```bash
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_USE_DEEP_GEMM=0
export VLLM_FLASHINFER_MOE_BACKEND=latency

vllm serve internlm/Intern-S2-Preview-397B \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --mm-encoder-tp-mode data \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

- Long-Context Serving

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve internlm/Intern-S2-Preview-397B \
  --tensor-parallel-size 8 \
  --max-model-len 1010000 \
  --reasoning-parser qwen3 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

### SGLang (>=v0.5.13)

- Basic Serving Without MTP

```bash
python3 -m sglang.launch_server \
    --model-path internlm/Intern-S2-Preview-397B \
    --trust-remote-code \
    --tp-size 8 \
    --mem-fraction-static 0.8 \
    --enable-flashinfer-allreduce-fusion \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

- Serving With MTP

```bash
SGLANG_ENABLE_SPEC_V2=1 \
python3 -m sglang.launch_server \
  --model-path internlm/Intern-S2-Preview-397B \
  --trust-remote-code \
  --tp-size 8 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mem-fraction-static 0.8 \
  --mamba-scheduler-strategy extra_buffer \
  --enable-flashinfer-allreduce-fusion \
  --speculative-algo 'NEXTN' \
  --speculative-eagle-topk 1 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4
```

## Agent Integration

Intern-S2-Preview-397B can be plugged into agent frameworks in two ways:

- Connecting to a self-hosted deployment
- Calling the official Intern API

The examples below cover OpenAI-compatible agent frameworks such as OpenClaw and Hermes, and Claude Code.

### Self-Hosted Deployment

First, serve the model with LMDeploy. The examples below assume the server is running at `http://0.0.0.0:23333`.

When launching LMDeploy for tool-calling workloads, remember to set `--tool-call-parser interns2-preview` so tool calls are parsed correctly.

#### Connecting Agent Frameworks

Most agent frameworks accept an OpenAI-compatible endpoint. Point the framework to the LMDeploy server base URL:

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://0.0.0.0:23333/v1
export OPENAI_MODEL=internlm/Intern-S2-Preview-397B
```

You can verify the connection with the following request:

```bash
curl http://0.0.0.0:23333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "internlm/Intern-S2-Preview-397B",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.8,
    "top_p": 0.95
  }'
```

#### Connecting Claude Code

LMDeploy exposes an Anthropic-compatible `/v1/messages` endpoint that Claude Code can talk to directly. Add the following configuration to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:23333",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_MODEL": "internlm/Intern-S2-Preview-397B",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "internlm/Intern-S2-Preview-397B"
  }
}
```

For a full walkthrough including curl verification, model routing, and troubleshooting, see [LMDeploy Claude Code Integration](https://lmdeploy.readthedocs.io/en/latest/intergration/claude_code.html).

### Official Intern API

If you do not want to self-host Intern-S2-Preview-397B, you can use the official Intern API. Register at [internlm.intern-ai.org.cn](https://internlm.intern-ai.org.cn/) and create an API token such as `sk-xxxxxxxx`.

#### Connecting Agent Frameworks

The service is OpenAI-compatible, so agent frameworks can use the official endpoint directly. Set the base URL to `https://chat.intern-ai.org.cn/api/v1` and the model name to `intern-s2-preview-397b`.

```bash
export OPENAI_API_KEY=sk-xxxxxxxx
export OPENAI_BASE_URL=https://chat.intern-ai.org.cn/api/v1
export OPENAI_MODEL=intern-s2-preview-397b
```

You can verify the connection with the following request:

```bash
curl https://chat.intern-ai.org.cn/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxxxxxxx" \
  -d '{
    "model": "intern-s2-preview-397b",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.8,
    "top_p": 0.95
  }'
```

Refer to the [Intern API documentation](https://internlm.intern-ai.org.cn/api/document?lang=en) for the current endpoint, available model names, rate limits, and advanced parameters.

#### Connecting Claude Code

Claude Code can route to the official Intern API by pointing `ANTHROPIC_BASE_URL` at the Intern Anthropic-compatible gateway:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://chat.intern-ai.org.cn",
    "ANTHROPIC_AUTH_TOKEN": "your-api-token",
    "ANTHROPIC_MODEL": "intern-s2-preview-397b",
    "ANTHROPIC_SMALL_FAST_MODEL": "intern-s2-preview-397b"
  }
}
```

Then start Claude Code with the following command:

```bash
claude --model intern-s2-preview-397b
```

For step-by-step setup, see [Intern API Claude Code Integration](https://internlm.intern-ai.org.cn/docEn/docs/Claude-Code-Integration).
