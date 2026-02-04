# Intern-S1-Pro 使用超参

## 采样超参

我们推荐使用如下的超参数以获得更好的生成效果：

```python
top_p = 0.95
top_k = 50
min_p = 0.0
temperature = 0.8
```

## 部署服务

Intern-S1-Pro 模型为万亿参数规模模型，采用FP8精度格式存储。该模型部署需至少两台搭载8张H200显卡的计算节点，可选用以下任意一种部署配置：

- 张量并行（TP）
- 数据并行（DP）+ 专家并行（EP）

> 注：本指南中的部署示例仅供参考，并非最新或最优配置方案。推理框架仍在持续开发迭代中——请务必参考各框架维护方发布的官方文档，以保障模型运行的最佳性能与兼容性。

### LMDeploy

依赖版本 `lmdeploy>=0.12.0`

- 张量并行（TP）

```bash
# start ray on node 0 and node 1

# node 0
lmdeploy serve api_server internlm/Intern-S1-Pro --backend pytorch --tp 16
```

- 数据并行（DP）+ 专家并行（EP）

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

- 张量并行（TP） + 专家并行（EP）

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

- 数据并行（DP） + 专家并行（EP）

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

- 张量并行（TP） + 专家并行（EP）

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

### 切换深度思考模式与非思考模式

Intern-S1-Pro 默认启用“深度思考模式（thinking mode）”，该模式可增强模型的推理能力，从而生成更高质量的回复。若希望关闭此功能，只需在 `tokenizer.apply_chat_template` 中设置参数 `enable_thinking=False` 即可。

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # think mode indicator
)
```

部署 Intern-S1 模型时，你可以通过调整请求中的 `enable_thinking` 参数来动态控制深度思考模式的开启与关闭。

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
