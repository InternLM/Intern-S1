## Fine-tuning

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) supports fine-tuning Intern-s1-mini Model. See this [PR](https://github.com/hiyouga/LLaMA-Factory/pull/8976)

### lora sft

Create a new file `examples/train_full/interns1_mini_lora_sft.yaml` with the following content:

```yaml
### model
model_name_or_path:  internlm/Intern-S1-mini
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
lora_rank: 8
lora_target: all

### dataset
dataset: mllm_demo,identity,alpaca_en_demo
template: intern_s1
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/interns1_mini/full/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
# 1 gpu >=22g
```

Run this command:

```bash
CUDA_VISIBLE_DEVICES=0  DISABLE_VERSION_CHECK=1 lamafactory-cli train examples/train_full/interns1_lora_sft.yaml
```

### full sft

Create a new file `examples/train_full/interns1_mini_full_sft.yaml` with the following content:

```yaml
### model
model_name_or_path: internlm/Intern-S1-mini
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: mllm_demo,identity,alpaca_en_demo
template: intern_s1
cutoff_len: 2048
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/interns1_mini/full/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
# val_size: 0.1
# per_device_eval_batch_size: 1
# eval_strategy: steps
# eval_steps: 500
```

Run this command:

```bash
DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_full/interns1_mini_full_sft.yaml
# or
DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/interns1_mini_full_sft.yaml
```

Note: `pip install transformers>=4.55.2`
