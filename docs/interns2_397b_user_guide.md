# Intern-S2-397B User Guide

## Sampling Parameters

We recommend using the following hyperparameters to ensure better results.

```python
top_p = 0.95
top_k = 50
min_p = 0.0
temperature = 0.8
```

## Serving

We recommend deploying Intern-S2-397B on H100 (x8) or H200 (x8) nodes. This guide provides deployment examples for the following configurations:

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

- Serving With MTP

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

- Long-Context Serving

For long-context inference, configure both `--session-len` and YaRN RoPE parameters. The following example uses a 512k context length:

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

### vLLM (>=v0.22.1)

- Basic Serving Without MTP

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

- Serving With MTP

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

- Long-Context Serving

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve internlm/Intern-S2-397B \
  --tensor-parallel-size 8 \
  --max-model-len 1010000 \
  --reasoning-parser qwen3 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

### SGLang (>=v0.5.13)

- Basic Serving Without MTP

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

- Serving With MTP

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

## Advanced Usage

### Tool Calling

Tool Calling lets the model extend its capabilities by invoking external tools and APIs. The example below shows how to use it to fetch the latest weather forecast via an OpenAI-compatible API (based on lmdeploy api server).

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

### Switching Between Thinking and Non-Thinking Modes

Intern-S2-397B enables thinking mode by default, enhancing the model's reasoning capabilities to generate higher-quality responses. This feature can be disabled by setting `enable_thinking=False` in `tokenizer.apply_chat_template`

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # think mode indicator
)
```

When serving Intern-S2-397B models, you can dynamically control the thinking mode by adjusting the `enable_thinking` parameter in your requests.

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

> Note: We do not recommend disabling thinking mode for agentic tasks.

## Time Series Demo

Time series inference is currently only supported in LMDeploy. To get started, deploy Intern-S2-397B with LMDeploy following the [Serving](#serving) section above.
Below is an example of detecting earthquake events from a time series signal file. Additional data types and functionalities are also supported.

**Please note**: in the message content, the order of time_series_url and the text prompt can be arbitrary.

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

For time series forecasting, `forecast_horizon` is optional. Set it to an integer to produce a forecast of exactly that length, or set it to `None` to let the model infer the horizon from the text prompt.

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

## Agent Integration

Intern-S2-397B can be plugged into agent frameworks in two ways:

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
export OPENAI_MODEL=internlm/Intern-S2-397B
```

You can verify the connection with the following request:

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

#### Connecting Claude Code

LMDeploy exposes an Anthropic-compatible `/v1/messages` endpoint that Claude Code can talk to directly. Add the following configuration to `~/.claude/settings.json`:

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

For a full walkthrough including curl verification, model routing, and troubleshooting, see [LMDeploy Claude Code Integration](https://lmdeploy.readthedocs.io/en/latest/intergration/claude_code.html).

### Official Intern API

If you do not want to self-host Intern-S2-397B, you can use the official Intern API. Register at [internlm.intern-ai.org.cn](https://internlm.intern-ai.org.cn/) and create an API token such as `sk-xxxxxxxx`.

#### Connecting Agent Frameworks

The service is OpenAI-compatible, so agent frameworks can use the official endpoint directly. Set the base URL to `https://chat.intern-ai.org.cn/api/v1` and the model name to `intern-s2-397b`.

```bash
export OPENAI_API_KEY=sk-xxxxxxxx
export OPENAI_BASE_URL=https://chat.intern-ai.org.cn/api/v1
export OPENAI_MODEL=intern-s2-397b
```

You can verify the connection with the following request:

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

Refer to the [Intern API documentation](https://internlm.intern-ai.org.cn/api/document?lang=en) for the current endpoint, available model names, rate limits, and advanced parameters.

#### Connecting Claude Code

Claude Code can route to the official Intern API by pointing `ANTHROPIC_BASE_URL` at the Intern Anthropic-compatible gateway:

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

Then start Claude Code with the following command:

```bash
claude --model intern-s2-397b
```

For step-by-step setup, see [Intern API Claude Code Integration](https://internlm.intern-ai.org.cn/docEn/docs/Claude-Code-Integration).
