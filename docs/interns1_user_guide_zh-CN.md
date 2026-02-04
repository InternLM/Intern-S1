# InternS1 使用指南

### 采样参数建议

我们推荐使用如下的超参数以获得更好的生成效果：

Intern-S1:

```python
top_p = 1.0
top_k = 50
min_p = 0.0
temperature = 0.7
```

Intern-S1-mini:

```python
top_p = 1.0
top_k = 50
min_p = 0.0
temperature = 0.8
```

### Transformers 示例

下面是使用文本和多模态输入进行推理生成的示例代码。

> **请使用 transformers >= 4.53.0 的版本以确保模型正常运行。**

#### 文本输入

```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model_name = "internlm/Intern-S1"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "tell me about an interesting physical phenomenon."},
        ],
    }
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

generate_ids = model.generate(**inputs, max_new_tokens=32768)
decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(decoded_output)
```

#### 图像输入

```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model_name = "internlm/Intern-S1"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "Please describe the image explicitly."},
        ],
    }
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

generate_ids = model.generate(**inputs, max_new_tokens=32768)
decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(decoded_output)
```

#### 视频输入

请确保安装了 decord 视频解码库，可通过 `pip install decord` 安装。

```python
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model_name = "internlm/Intern-S1"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
                },
                {"type": "text", "text": "What type of shot is the man performing?"},
            ],
        }
    ]

inputs = processor.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        video_load_backend="decord",
        tokenize=True,
        return_dict=True,
    ).to(model.device, dtype=torch.float16)

generate_ids = model.generate(**inputs, max_new_tokens=32768)
decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(decoded_output)
```

### 部署服务

在部署 InternS1 系列模型时，对于硬件的最低要求如下表所示：

|                                       Model                                       | A100(GPUs) | H800(GPUs) | H100(GPUs) | H200(GPUs) |
| :-------------------------------------------------------------------------------: | :--------: | :--------: | :--------: | :--------: |
|          [internlm/Intern-S1](https://huggingface.co/internlm/Intern-S1)          |     8      |     8      |     8      |     4      |
|      [internlm/Intern-S1-FP8](https://huggingface.co/internlm/Intern-S1-FP8)      |     -      |     4      |     4      |     2      |
|     [internlm/Intern-S1-mini](https://huggingface.co/internlm/Intern-S1-mini)     |     1      |     1      |     1      |     1      |
| [internlm/Intern-S1-mini-FP8](https://huggingface.co/internlm/Intern-S1-mini-FP8) |     -      |     1      |     1      |     1      |

你可以使用以下这些 LLM 推理引擎来创建一个兼容 OpenAI 协议的服务:

#### [lmdeploy (>=0.9.2)](https://github.com/InternLM/lmdeploy)

```bash
lmdeploy serve api_server internlm/Intern-S1 --reasoning-parser intern-s1 --tool-call-parser intern-s1 --tp 8
```

#### [vllm (>=0.10.1)](https://github.com/vllm-project/vllm)

```bash
vllm serve internlm/Intern-S1 --tensor-parallel-size 8 --trust-remote-code
```

#### [sglang](https://github.com/sgl-project/sglang)

```bash
python3 -m sglang.launch_server \
    --model-path internlm/Intern-S1 \
    --trust-remote-code \
    --tp 8 \
    --grammar-backend none
```

#### ollama

```bash
# install ollama
curl -fsSL https://ollama.com/install.sh | sh
# fetch model
ollama pull internlm/interns1
# run model
ollama run internlm/interns1
# then use openai client to call on http://localhost:11434/v1
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
    temperature=0.8,
    top_p=0.8,
    stream=False,
    extra_body=dict(spaces_between_special_tokens=False, enable_thinking=False),
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
    top_p=0.8,
    stream=False,
    extra_body=dict(spaces_between_special_tokens=False, enable_thinking=False),
    tools=tools)
print(response.choices[0].message.content)
```

### 切换深度思考模式与非思考模式

Intern-S1 默认启用“深度思考模式（thinking mode）”，该模式可增强模型的推理能力，从而生成更高质量的回复。若希望关闭此功能，只需在 `tokenizer.apply_chat_template` 中设置参数 `enable_thinking=False` 即可。

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # think mode indicator
)
```

使用 **LMDeploy** 部署 Intern-S1 模型时，你可以通过调整请求中的 `enable_thinking` 参数来动态控制深度思考模式的开启与关闭。

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
    temperature=0.7,
    top_p=0.8,
    max_tokens=2048,
    extra_body={
        "enable_thinking": False,
    }
)
print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
```

对于 **vllm** 和 **sglang** 用户，可通过以下方式进行配置：

```python
extra_body={
    "chat_template_kwargs": {"enable_thinking": False}
}
```
