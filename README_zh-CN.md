## 书生 Intern-S1

<div align="center">
<img src="./assets/title_zh.jpg" />

<div>&nbsp;</div>

[🤗Huggingface](https://huggingface.co/collections/internlm/intern-s1-6882e325e8ac1c58ba108aa5) •  [<img src="./assets/modelscope_logo.png" width="20px" /> ModelScope](https://modelscope.cn/collections/Intern-S1-29b3100f15e240) • [📜技术报告 (即将公开)](<>) • [💬在线体验](https://chat.intern-ai.org.cn/)

[English](./README.md) |
[简体中文](./README_zh-CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://cdn.vansin.top/intern-s1.jpg" target="_blank">WeChat</a>
</p>

## 简介

我们推出了 **Intern-S1**，这是我们推出的**最先进的开源多模态推理模型**。Intern-S1 在具备强大通用任务能力的同时，在广泛的**科学任务中也达到了最先进的性能**，可与最先进的闭源商业模型相媲美。

Intern-S1 基于一个 235B 的 MoE 语言模型 (Qwen3) 和一个 6B 的视觉编码器 (InternViT) 构建，并在 **5T token** 的多模态数据上进行了续训，其中包含**超过 2.5T 的科学领域 token**。这一训练策略使得该模型不仅保留了强大的通用能力，还在专业科学任务上表现出色，例如**解析化学结构、理解蛋白质序列、规划化合物合成路径**，使 Intern-S1 成为了能够应对真实科研任务的 AI 助手。

另外，我们还推出了 **Intern-S1-mini**，这是一个使用了 Intern-S1 同样训练技术的轻量级模型，包含一个 8B 的语言模型和一个 400M 的视觉编码器。

### 特性

- 在语言与视觉推理基准测试中表现强劲，尤其擅长科学任务。

- 在包含超过 50% 科学专业数据的 5T 规模数据集上持续预训练，深度融合专业领域知识。

- 动态分词器原生支持对分子式、蛋白质序列、地震信号等数据的理解。

## 模型库

### Intern-S1

|                                                                    | BF16                                                                                              | FP8                                                                                                       | GGUF                                                                                                        |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 🤗HuggingFace                                                      | [internlm/Intern-S1](https://huggingface.co/internlm/Intern-S1)                                   | [internlm/Intern-S1-FP8](https://huggingface.co/internlm/Intern-S1-FP8)                                   | [internlm/Intern-S1-GGUF](https://huggingface.co/internlm/Intern-S1-GGUF)                                   |
| <img src="./assets/modelscope_logo.png" width="20px" /> ModelScope | [Shanghai_AI_Laboratory/Intern-S1](https://modelscope.cn/models/Shanghai_AI_Laboratory/Intern-S1) | [Shanghai_AI_Laboratory/Intern-S1-FP8](https://modelscope.cn/models/Shanghai_AI_Laboratory/Intern-S1-FP8) | [Shanghai_AI_Laboratory/Intern-S1-GGUF](https://modelscope.cn/models/Shanghai_AI_Laboratory/Intern-S1-GGUF) |

### Intern-S1-mini

|                                                                    | BF16                                                                                                        | FP8                                                                                                                 | GGUF                                                                                |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| 🤗HuggingFace                                                      | [internlm/Intern-S1-mini](https://huggingface.co/internlm/Intern-S1-mini)                                   | [internlm/Intern-S1-mini-FP8](https://huggingface.co/internlm/Intern-S1-mini-FP8)                                   | [internlm/Intern-S1-mini-GGUF](https://huggingface.co/internlm/Intern-S1-mini-GGUF) |
| <img src="./assets/modelscope_logo.png" width="20px" /> ModelScope | [Shanghai_AI_Laboratory/Intern-S1-mini](https://modelscope.cn/models/Shanghai_AI_Laboratory/Intern-S1-mini) | [Shanghai_AI_Laboratory/Intern-S1-mini-FP8](https://modelscope.cn/models/Shanghai_AI_Laboratory/Intern-S1-mini-FP8) | -                                                                                   |

## 性能评估

我们在多个通用数据集和科学数据集上评估了 Intern-S1 的表现，并与近期的视觉语言模型（VLMs）和大语言模型（LLMs）进行了对比，结果如下表所示。

### Intern-S1

<table>
  <thead>
    <tr>
      <th rowspan="2">评测集</th>
      <th colspan="2">Intern-S1</th>
      <th>InternVL3-78B</th>
      <th>Qwen2.5-VL-72B</th>
      <th>DS-R1-0528</th>
      <th>Qwen3-235B-A22B</th>
      <th>Kimi-K2-Instruct</th>
      <th>Gemini-2.5 Pro</th>
      <th>o3</th>
      <th>Grok-4</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>MMLU-Pro</td><td colspan="2">83.5 ✅</td><td>73.0</td><td>72.1</td><td>83.4</td><td>82.2</td><td>82.7</td><td>86.0</td><td>85.0</td><td>85.9</td></tr>
    <tr><td>MMMU</td><td colspan="2">77.7 ✅</td><td>72.2</td><td>70.2</td><td>-</td><td>-</td><td>-</td><td>81.9</td><td>80.8</td><td>77.9</td></tr>
    <tr><td>GPQA</td><td colspan="2">77.3</td><td>49.9</td><td>49.0</td><td>80.6</td><td>71.1</td><td>77.8</td><td>83.8</td><td>83.3</td><td>87.5</td></tr>
    <tr><td>MMStar</td><td colspan="2">74.9 ✅</td><td>72.5</td><td>70.8</td><td>-</td><td>-</td><td>-</td><td>79.3</td><td>75.1</td><td>69.6</td></tr>
    <tr><td>MathVista</td><td colspan="2">81.5 👑</td><td>79.0</td><td>74.8</td><td>-</td><td>-</td><td>-</td><td>80.3</td><td>77.5</td><td>72.5</td></tr>
    <tr><td>AIME2025</td><td colspan="2">86.0</td><td>10.7</td><td>10.9</td><td>87.5</td><td>81.5</td><td>51.4</td><td>83.0</td><td>88.9</td><td>91.7</td></tr>
    <tr><td>MathVision</td><td colspan="2">62.5 ✅</td><td>43.1</td><td>38.1</td><td>-</td><td>-</td><td>-</td><td>73.0</td><td>67.7</td><td>67.3</td></tr>
    <tr><td>IFEval</td><td colspan="2">86.7</td><td>75.6</td><td>83.9</td><td>79.7</td><td>85.0</td><td>90.2</td><td>91.5</td><td>92.2</td><td>92.8</td></tr>
    <tr><td>SFE</td><td colspan="2">44.3 👑</td><td>36.2</td><td>30.5</td><td>-</td><td>-</td><td>-</td><td>43.0</td><td>37.7</td><td>31.2</td></tr>
    <tr><td>Physics</td><td colspan="2">44.0 ✅</td><td>23.1</td><td>15.7</td><td>-</td><td>-</td><td>-</td><td>40.0</td><td>47.9</td><td>42.8</td></tr>
    <tr><td>SmolInstruct</td><td colspan="2">51.0 👑</td><td>19.4</td><td>21.0</td><td>30.7</td><td>28.7</td><td>48.1</td><td>40.4</td><td>43.9</td><td>47.3</td></tr>
    <tr><td>ChemBench</td><td colspan="2">83.4 👑</td><td>61.3</td><td>61.6</td><td>75.6</td><td>75.8</td><td>75.3</td><td>82.8</td><td>81.6</td><td>83.3</td></tr>
    <tr><td>MatBench</td><td colspan="2">75.0 👑</td><td>49.3</td><td>51.5</td><td>57.7</td><td>52.1</td><td>61.7</td><td>61.7</td><td>61.6</td><td>67.9</td></tr>
    <tr><td>MicroVQA</td><td colspan="2">63.9 👑</td><td>59.1</td><td>53.0</td><td>-</td><td>-</td><td>-</td><td>63.1</td><td>58.3</td><td>59.5</td></tr>
    <tr><td>ProteinLMBench</td><td colspan="2">63.1</td><td>61.6</td><td>61.0</td><td>61.4</td><td>59.8</td><td>66.7</td><td>62.9</td><td>67.7</td><td>66.2</td></tr>
    <tr><td>MSEarthMCQ</td><td colspan="2">65.7 👑</td><td>57.2</td><td>37.6</td><td>-</td><td>-</td><td>-</td><td>59.9</td><td>61.0</td><td>58.0</td></tr>
    <tr><td>XLRS-Bench</td><td colspan="2">55.0 👑</td><td>49.3</td><td>50.9</td><td>-</td><td>-</td><td>-</td><td>45.2</td><td>43.6</td><td>45.4</td></tr>
  </tbody>
</table>

> **注意**: ✅ 表示在开源模型中取得最优， 👑 表示在所有模型中取得最优。

### Intern-S1-mini

| 评测集         | Intern-S1-mini | Qwen3-8B | GLM-4.1V | MiMo-VL-7B-RL-2508 |
| -------------- | -------------- | -------- | -------- | ------------------ |
| MMLU-Pro       | **74.78**      | 73.7     | 57.1     | 73.93              |
| MMMU           | **72.33**      | N/A      | 69.9     | 70.4               |
| MMStar         | 65.2           | N/A      | 71.5     | 72.9               |
| GPQA           | **65.15**      | 62       | 50.32    | 60.35              |
| AIME2024       | **84.58**      | 76       | 36.2     | 72.6               |
| AIME2025       | **80**         | 67.3     | 32       | 64.4               |
| MathVision     | 51.41          | N/A      | 53.9     | 54.5               |
| MathVista      | 70.3           | N/A      | 80.7     | 79.4               |
| IFEval         | 81.15          | 85       | 71.53    | 71.4               |
| SFE            | 35.84          | N/A      | 43.2     | 43.9               |
| Physics        | **28.76**      | N/A      | 4.3      | 23.9               |
| SmolInstruct   | **32.2**       | 17.6     | 18.1     | 16.11              |
| ChemBench      | **76.47**      | 61.1     | 56.2     | 66.78              |
| MatBench       | **61.55**      | 45.24    | 54.3     | 46.9               |
| MicroVQA       | **56.62**      | N/A      | 50.2     | 50.96              |
| ProteinLMBench | 58.47          | 59.1     | 58.3     | 59.8               |
| MSEarthMCQ     | **58.12**      | N/A      | 50.3     | 47.3               |
| XLRS-Bench     | **51.63**      | N/A      | 49.8     | 12.29              |

评估使用了 [OpenCompass](https://github.com/open-compass/OpenCompass/) 和 [VLMEvalkit](https://github.com/open-compass/vlmevalkit)。

## 快速开始

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

#### [lmdeploy(>=0.9.2)](https://github.com/InternLM/lmdeploy)

```bash
lmdeploy serve api_server internlm/Intern-S1 --reasoning-parser intern-s1 --tool-call-parser intern-s1 --tp 8
```

#### [vllm](https://github.com/vllm-project/vllm)

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

## 开源协议

该项目使用 Apache 2.0 开源协议。
