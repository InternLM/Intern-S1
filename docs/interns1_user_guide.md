# InternS1 User Guide

### Sampling Parameters

We recommend using the following hyperparameters to ensure better results

For Intern-S1:

```python
top_p = 1.0
top_k = 50
min_p = 0.0
temperature = 0.7
```

For Intern-S1-mini:

```python
top_p = 1.0
top_k = 50
min_p = 0.0
temperature = 0.8
```

### Transformers

The following provides demo code illustrating how to generate based on text and multimodal inputs.

> **Please use transformers>=4.53.0 to ensure the model works normally.**

#### Text input

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

#### Image input

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

#### Video input

Please ensure that the decord video decoding library is installed via `pip install decord`.

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

### Serving

The minimum hardware requirements for deploying Intern-S1 series models are:

|                                       Model                                       | A100(GPUs) | H800(GPUs) | H100(GPUs) | H200(GPUs) |
| :-------------------------------------------------------------------------------: | :--------: | :--------: | :--------: | :--------: |
|          [internlm/Intern-S1](https://huggingface.co/internlm/Intern-S1)          |     8      |     8      |     8      |     4      |
|      [internlm/Intern-S1-FP8](https://huggingface.co/internlm/Intern-S1-FP8)      |     -      |     4      |     4      |     2      |
|     [internlm/Intern-S1-mini](https://huggingface.co/internlm/Intern-S1-mini)     |     1      |     1      |     1      |     1      |
| [internlm/Intern-S1-mini-FP8](https://huggingface.co/internlm/Intern-S1-mini-FP8) |     -      |     1      |     1      |     1      |

You can utilize one of the following LLM inference frameworks to create an OpenAI compatible server:

#### [lmdeploy (>=0.9.2.post1)](https://github.com/InternLM/lmdeploy)

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

#### ollama for local deployment:

```bash
# install ollama
curl -fsSL https://ollama.com/install.sh | sh
# fetch model
ollama pull internlm/interns1
# run model
ollama run internlm/interns1
# then use openai client to call on http://localhost:11434/v1
```

## Advanced Usage

### Tool Calling

Many Large Language Models (LLMs) now feature **Tool Calling**, a powerful capability that allows them to extend their functionality by interacting with external tools and APIs. This enables models to perform tasks like fetching up-to-the-minute information, running code, or calling functions within other applications.

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

### Switching Between Thinking and Non-Thinking Modes

Intern-S1 enables thinking mode by default, enhancing the model's reasoning capabilities to generate higher-quality responses. This feature can be disabled by setting `enable_thinking=False` in `tokenizer.apply_chat_template`

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # think mode indicator
)
```

With LMDeploy serving Intern-S1 models, you can dynamically control the thinking mode by adjusting the `enable_thinking` parameter in your requests.

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

For vllm and sglang users, configure this through,

```python
extra_body={
    "chat_template_kwargs": {"enable_thinking": false}
}
```
