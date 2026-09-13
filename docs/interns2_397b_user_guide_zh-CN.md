# Intern-S2-397B 使用指南

## 采样超参

我们推荐使用以下超参数以获得更好的生成效果：

```python
top_p = 0.95
top_k = 50
min_p = 0.0
temperature = 0.8
```

## 部署服务

我们推荐在 H100（x8）或 H200（x8）节点上部署 Intern-S2-397B。本指南提供以下几类部署配置示例：

- 不启用 MTP 的基础服务
- MTP 投机解码
- 结合 YaRN RoPE 配置的长上下文推理

> 注：本指南中的部署示例仅供参考，并非最新或最优配置方案。推理框架仍在持续开发迭代中，请结合各框架维护方发布的官方文档和本地验证结果调整生产部署配置。

### LMDeploy（>=0.14.0）

- 不启用 MTP 的基础服务

```bash
# proxy server
lmdeploy serve proxy --server-name ${proxy_server_ip} --server-port ${proxy_server_port}

# api_server
lmdeploy serve api_server \
    internlm/Intern-S2-397B \
    --trust-remote-code \
    --backend pytorch \
    --dp 4 \
    --ep 8 \
    --enable-prefix-caching \
    --proxy-url http://${proxy_server_ip}:${proxy_server_port} \
    --reasoning-parser default \
    --tool-call-parser interns2-preview
```

- 启用 MTP 的服务

```bash
lmdeploy serve api_server \
    internlm/Intern-S2-397B \
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

- 长上下文服务

进行长上下文推理时，需要同时配置 `--session-len` 和 YaRN RoPE 参数。以下示例使用 512k 上下文长度：

```bash
lmdeploy serve api_server \
    internlm/Intern-S2-397B \
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

### vLLM（>=v0.22.1）

- 不启用 MTP 的基础服务

```bash
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_USE_DEEP_GEMM=0
export VLLM_FLASHINFER_MOE_BACKEND=latency

vllm serve internlm/Intern-S2-397B \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data
```

- 启用 MTP 的服务

```bash
export VLLM_DEEP_GEMM_WARMUP=skip
export VLLM_USE_DEEP_GEMM=0
export VLLM_FLASHINFER_MOE_BACKEND=latency

vllm serve internlm/Intern-S2-397B \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --mm-encoder-tp-mode data \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

- 长上下文服务

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve internlm/Intern-S2-397B \
  --tensor-parallel-size 8 \
  --max-model-len 1010000 \
  --reasoning-parser qwen3 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

### SGLang（>=v0.5.13）

- 不启用 MTP 的基础服务

```bash
python3 -m sglang.launch_server \
    --model-path internlm/Intern-S2-397B \
    --trust-remote-code \
    --tp-size 8 \
    --mem-fraction-static 0.8 \
    --enable-flashinfer-allreduce-fusion \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

- 启用 MTP 的服务

```bash
SGLANG_ENABLE_SPEC_V2=1 \
python3 -m sglang.launch_server \
  --model-path internlm/Intern-S2-397B \
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

## 高级用法

### 工具调用（Tool Calling）

许多大型语言模型现在具备了 **工具调用（Tool Calling）** 的能力，使它们能够通过与外部工具和 API 的交互来扩展自身的能力。这使得模型可以执行如获取最新信息、运行代码，或调用其他应用程序中的函数等任务。

对开发者来说，越来越多的开源语言模型设计为兼容 OpenAI API。这意味着你可以复用 OpenAI 的接口，在这些开源模型中实现工具调用。因此，本教程中演示的代码具有高度的通用性——不仅适用于 OpenAI 的模型，也适用于任何遵循相同接口标准的模型。

下面我们通过一个实际的代码示例，演示如何使用工具调用功能来获取最新的天气预报（基于 lmdeploy api server）。

```python


from openai import OpenAI
import json


def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit,
    }


def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit,
    }

def get_function_by_name(name):
    if name == "get_current_temperature":
        return get_current_temperature
    if name == "get_temperature_date":
        return get_temperature_date

tools = [{
    'type': 'function',
    'function': {
        'name': 'get_current_temperature',
        'description': 'Get current temperature at a location.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The location to get the temperature for, in the format \'City, State, Country\'.'
                },
                'unit': {
                    'type': 'string',
                    'enum': [
                        'celsius',
                        'fahrenheit'
                    ],
                    'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                }
            },
            'required': [
                'location'
            ]
        }
    }
}, {
    'type': 'function',
    'function': {
        'name': 'get_temperature_date',
        'description': 'Get temperature at a location and date.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The location to get the temperature for, in the format \'City, State, Country\'.'
                },
                'date': {
                    'type': 'string',
                    'description': 'The date to get the temperature for, in the format \'Year-Month-Day\'.'
                },
                'unit': {
                    'type': 'string',
                    'enum': [
                        'celsius',
                        'fahrenheit'
                    ],
                    'description': 'The unit to return the temperature in. Defaults to \'celsius\'.'
                }
            },
            'required': [
                'location',
                'date'
            ]
        }
    }
}]



messages = [
    {'role': 'user', 'content': 'Today is 2024-11-14, What\'s the temperature in San Francisco now? How about tomorrow?'}
]

openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:23333/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_name = "internlm/Intern-S2-397B"  # Must match the model ID served by your deployment.
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    max_tokens=32768,
    temperature=0.8,
    top_p=0.95,
    extra_body=dict(spaces_between_special_tokens=False),
    tools=tools)
print(response.choices[0].message)
messages.append(response.choices[0].message)

for tool_call in response.choices[0].message.tool_calls:
    tool_call_args = json.loads(tool_call.function.arguments)
    tool_call_result = get_function_by_name(tool_call.function.name)(**tool_call_args)
    tool_call_result = json.dumps(tool_call_result, ensure_ascii=False)
    messages.append({
        'role': 'tool',
        'name': tool_call.function.name,
        'content': tool_call_result,
        'tool_call_id': tool_call.id
    })

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.95,
    extra_body=dict(spaces_between_special_tokens=False),
    tools=tools)
print(response.choices[0].message)
```

### 切换深度思考模式与非思考模式

Intern-S2-397B 默认启用“深度思考模式（thinking mode）”，该模式可增强模型的推理能力，从而生成更高质量的回复。若希望关闭此功能，只需在 `tokenizer.apply_chat_template` 中设置参数 `enable_thinking=False` 即可。

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # think mode indicator
)
```

在部署 Intern-S2-397B 提供服务时，你还可以通过在请求中调整 `enable_thinking` 参数来动态控制思考模式。

```python
from openai import OpenAI
import json

messages = [
{
    'role': 'user',
    'content': 'who are you'
}, {
    'role': 'assistant',
    'content': 'I am an AI'
}, {
    'role': 'user',
    'content': 'AGI is?'
}]

openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:23333/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_name = "internlm/Intern-S2-397B"  # Must match the model ID served by your deployment.

response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.8,
    top_p=0.95,
    max_tokens=2048,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    }
)
print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
```

> 注意：不建议在 agentic 任务中关闭深度思考模式。

## 时序数据示例

时序推理目前仅 LMDeploy 支持。开始之前，请先参考上文的 [部署服务](#部署服务) 章节使用 LMDeploy 部署 Intern-S2-397B。
下面是一个从时序信号文件中检测地震事件的示例，此外也支持更多数据类型与功能。

**请注意**：在 message 的 content 中，`time_series_url` 与文本 prompt 的先后顺序可以任意。

```python
from openai import OpenAI
from lmdeploy.vl.utils import encode_time_series_base64

openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model_name = "internlm/Intern-S2-397B"  # Must match the model ID served by your deployment.


def send_base64(file_path: str, sampling_rate: int = 100):
    """base64-encoded time-series data."""

    # encode_time_series_base64 accepts local file paths and http urls,
    # encoding time-series data (.npy, .csv, .wav, .mp3, .flac, etc.) into base64 strings.
    base64_ts = encode_time_series_base64(file_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "time_series_url",
                    "time_series_url": {
                        "url": f"data:time_series/npy;base64,{base64_ts}",
                        "sampling_rate": sampling_rate
                    },
                },
                {
                    "type": "text",
                    "text": "Please determine whether an Earthquake event has occurred in the provided time-series data. If so, please specify the starting time point indices of the P-wave and S-wave in the event."
                },
            ],
        }
    ]

    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=200,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )


def send_http_url(url: str, sampling_rate: int = 100):
    """http(s) url pointing to the time-series data."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "time_series_url",
                    "time_series_url": {
                        "url": url,
                        "sampling_rate": sampling_rate
                    },
                },
                {
                    "type": "text",
                    "text": "Please determine whether an Earthquake event has occurred in the provided time-series data. If so, please specify the starting time point indices of the P-wave and S-wave in the event."
                },
            ],
        }
    ]

    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=200,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )


def send_file_url(file_path: str, sampling_rate: int = 100):
    """file url pointing to the time-series data."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "time_series_url",
                    "time_series_url": {
                        "url": f"file://{file_path}",
                        "sampling_rate": sampling_rate
                    },
                },
                {
                    "type": "text",
                    "text": "Please determine whether an Earthquake event has occurred in the provided time-series data. If so, please specify the starting time point indices of the P-wave and S-wave in the event."
                },
            ],
        }
    ]

    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=200,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )

response = send_base64("./0092638_seism.npy")
# response = send_http_url("https://huggingface.co/internlm/Intern-S1-Pro/raw/main/0092638_seism.npy")
# response = send_file_url("./0092638_seism.npy")

print(response.choices[0].message)

```

对于时序预测任务，`forecast_horizon` 是可选参数。设置为整数时会生成恰好该长度的预测结果；设置为 `None` 时，则由模型根据文本 prompt 自行推断预测长度。

```python
def forecast_base64(file_path: str, forecast_horizon: int | None = None):
    base64_ts = encode_time_series_base64(file_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please complete a electric load forecasting task. "
                        "This dataset is based on historical electricity load data every half hour within 24 hours of the region, "
                        "as well as data on minimum temperature, maximum temperature, humidity, air pressure, etc., "
                        "to predict future load consumption every half hour within 24 hours. Here is the weather information for city TAS: "
                        "Historical date weather: minimum temperature of 279.71K, maximum temperature of 285.83K, humidity of 85.0%, "
                        "air pressure of 1003.0hPa. Forecast date weather: minimum temperature 280.54K, maximum temperature 286.47K, "
                        "humidity 74.0%, air pressure 1007.0hPa. This data has no relevant effect information. "
                        "Please predict the next 48 time points given information above."
                    ),
                },
                {
                    "type": "time_series_url",
                    "time_series_url": {
                        "url": f"data:time_series/npy;base64,{base64_ts}",
                    },
                },
            ],
        }
    ]

    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=16,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_forecasting": True,
            "forecast_horizon": forecast_horizon,
        },
    )


response = forecast_base64("./load_20210803_0.npy", forecast_horizon=None)
forecast = response.choices[0].message.ts_forecast
print("Point forecast:", forecast.point_forecast)
print("Quantile forecast:", forecast.quantile_forecast)
```

## Agent Framework 接入

Intern-S2-397B 可以通过两种方式接入 agent framework：

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
export OPENAI_MODEL=internlm/Intern-S2-397B
```

可以使用以下请求验证连接：

```bash
curl http://0.0.0.0:23333/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "internlm/Intern-S2-397B",
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
    "ANTHROPIC_MODEL": "internlm/Intern-S2-397B",
    "ANTHROPIC_CUSTOM_MODEL_OPTION": "internlm/Intern-S2-397B"
  }
}
```

完整的验证、模型路由和故障排查流程可参考 [LMDeploy Claude Code 接入文档](https://lmdeploy.readthedocs.io/en/latest/intergration/claude_code.html)。

### 官方 Intern API

如果不希望自部署 Intern-S2-397B，也可以使用官方 Intern API。请在 [internlm.intern-ai.org.cn](https://internlm.intern-ai.org.cn/) 注册并创建 API token，例如 `sk-xxxxxxxx`。

#### 接入 Agent Framework

官方服务兼容 OpenAI API，因此 agent framework 可以直接使用官方 endpoint。将 base URL 设置为 `https://chat.intern-ai.org.cn/api/v1`，模型名设置为 `intern-s2-397b`。

```bash
export OPENAI_API_KEY=sk-xxxxxxxx
export OPENAI_BASE_URL=https://chat.intern-ai.org.cn/api/v1
export OPENAI_MODEL=intern-s2-397b
```

可以使用以下请求验证连接：

```bash
curl https://chat.intern-ai.org.cn/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxxxxxxx" \
  -d '{
    "model": "intern-s2-397b",
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
    "ANTHROPIC_MODEL": "intern-s2-397b",
    "ANTHROPIC_SMALL_FAST_MODEL": "intern-s2-397b"
  }
}
```

随后使用以下命令启动 Claude Code：

```bash
claude --model intern-s2-397b
```

详细接入步骤请参考 [Intern API Claude Code 接入文档](https://internlm.intern-ai.org.cn/docEn/docs/Claude-Code-Integration)。
