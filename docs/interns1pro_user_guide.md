# Intern-S1-Pro User Guide

## Sampling Parameters

We recommend using the following hyperparameters to ensure better results

```python
top_p = 0.95
top_k = 50
min_p = 0.0
temperature = 0.8
```

## Serving

The Intern-S1-Pro release is a 1T parameter model stored in FP8 format. Deployment requires at least two 8-GPU H200 nodes, with either of the following configurations:

- Tensor Parallelism (TP)
- Data Parallelism (DP) + Expert Parallelism (EP)

> NOTE: The deployment examples in this guide are provided for reference only and may not represent the latest or most optimized configurations. Inference frameworks are under active development — always consult the official documentation from each framework’s maintainers to ensure peak performance and compatibility.

### LMDeploy

Required version `lmdeploy>=0.12.0`

- Tensor Parallelism

```bash
# start ray on node 0 and node 1

# node 0
lmdeploy serve api_server internlm/Intern-S1-Pro --backend pytorch --tp 16
```

- Data Parallelism + Expert Parallelism

```
# node 0, proxy server
lmdeploy serve proxy --server-name ${proxy_server_ip} --server-port ${proxy_server_port} --routing-strategy 'min_expected_latency' --serving-strategy Hybrid

# node 0
export LMDEPLOY_DP_MASTER_ADDR=${node0_ip}
export LMDEPLOY_DP_MASTER_PORT=29555
lmdeploy serve api_server \
    internlm/Intern-S1-Pro \
    --backend pytorch \
    --tp 1 \
    --dp 16 \
    --ep 16 \
    --proxy-url http://${proxy_server_ip}:${proxy_server_port} \
    --nnodes 2 \
    --node-rank 0 \
    --reasoning-parser intern-s1 \
    --tool-call-parser qwen3

# node 1
export LMDEPLOY_DP_MASTER_ADDR=${node0_ip}
export LMDEPLOY_DP_MASTER_PORT=29555
lmdeploy serve api_server \
    internlm/Intern-S1-Pro \
    --backend pytorch \
    --tp 1 \
    --dp 16 \
    --ep 16 \
    --proxy-url http://${proxy_server_ip}:${proxy_server_port} \
    --nnodes 2 \
    --node-rank 1 \
    --reasoning-parser intern-s1 \
    --tool-call-parser qwen3
```

### vLLM

- Tensor Parallelism + Expert Parallelism

```bash
# start ray on node 0 and node 1

# node 0
export VLLM_ENGINE_READY_TIMEOUT_S=10000
vllm serve internlm/Intern-S1-Pro \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --distributed-executor-backend ray \
    --max-model-len 65536 \
    --trust-remote-code \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

- Data Parallelism + Expert Parallelism

```bash
# node 0
export VLLM_ENGINE_READY_TIMEOUT_S=10000
vllm serve internlm/Intern-S1-Pro \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-address ${node0_ip} \
    --data-parallel-rpc-port 13345 \
    --gpu_memory_utilization 0.8 \
    --mm_processor_cache_gb=0 \
    --media-io-kwargs '{"video": {"num_frames": 768, "fps": 2}}' \
    --max-model-len 65536 \
    --trust-remote-code \
    --api-server-count=8 \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# node 1
export VLLM_ENGINE_READY_TIMEOUT_S=10000
vllm serve internlm/Intern-S1-Pro \
    --all2all-backend deepep_low_latency \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-start-rank 8 \
    --data-parallel-address ${node0_ip} \
    --data-parallel-rpc-port 13345 \
    --gpu_memory_utilization 0.8 \
    --mm_processor_cache_gb=0 \
    --media-io-kwargs '{"video": {"num_frames": 768, "fps": 2}}' \
    --max-model-len 65536 \
    --trust-remote-code \
    --headless \
    --reasoning-parser deepseek_r1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

> NOTE: To prevent out-of-memory (OOM) errors, we limit the context length using `--max-model-len 65536`. For datasets requiring longer responses, you may increase this value as needed. Additionally, video inference can consume substantial memory in vLLM API server processes; we therefore recommend setting `--media-io-kwargs '{"video": {"num_frames": 768, "fps": 2}}'` to constrain preprocessing memory usage during video benchmarking.

### SGLang

- Tensor Parallelism + Expert Parallelism

```bash
export DIST_ADDR=${master_node_ip}:${master_node_port}

# node 0
python3 -m sglang.launch_server \
  --model-path internlm/Intern-S1-Pro \
  --tp 16 \
  --ep 16 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --dist-init-addr ${DIST_ADDR} \
  --nnodes 2 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --keep-mm-feature-on-device \
  --node-rank 0 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen

# node 1
python3 -m sglang.launch_server \
  --model-path internlm/Intern-S1-Pro \
  --tp 16 \
  --ep 16 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --dist-init-addr ${DIST_ADDR} \
  --nnodes 2 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --keep-mm-feature-on-device \
  --node-rank 1 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen
```

## Advanced Usage

### Tool Calling

Many Large Language Models (LLMs) now feature Tool Calling, a powerful capability that allows them to extend their functionality by interacting with external tools and APIs. This enables models to perform tasks like fetching up-to-the-minute information, running code, or calling functions within other applications.

A key advantage for developers is that a growing number of open-source LLMs are designed to be compatible with the OpenAI API. This means you can leverage the same familiar syntax and structure from the OpenAI library to implement tool calling with these open-source models. As a result, the code demonstrated in this tutorial is versatile—it works not just with OpenAI models, but with any model that follows the same interface standard.

To illustrate how this works, let's dive into a practical code example that uses tool calling to get the latest weather forecast (based on lmdeploy api server).

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
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    max_tokens=32768,
    temperature=0.95,
    top_p=0.8,
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

Intern-S1-Pro enables thinking mode by default, enhancing the model's reasoning capabilities to generate higher-quality responses. This feature can be disabled by setting `enable_thinking=False` in `tokenizer.apply_chat_template`

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # think mode indicator
)
```

With serving Intern-S1-Pro models, you can dynamically control the thinking mode by adjusting the `enable_thinking` parameter in your requests.

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
model_name = client.models.list().data[0].id

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
