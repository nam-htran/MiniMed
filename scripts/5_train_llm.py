# scripts/5_train_llm.py
import yaml
import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, DPOTrainer, SFTConfig

def get_prompt_formatter(template_type):
    """Factory để lấy hàm format prompt dựa trên config."""
    if template_type == "medcot":
        return lambda ex: {"text": f"Question: {ex['question']}\nReasoning: {ex['medcot_cot']}\nAnswer: {ex['answer']}"}
    elif template_type == "default_cot":
        return lambda ex: {"text": f"Question: {ex['question']}\nReasoning: {ex['default_cot']}\nAnswer: {ex['answer']}"}
    elif template_type == "trm":
        return lambda ex: {
            "text": f"<bos><start_of_turn>user\nYou are a medical reasoning engine. Your task is to self-enhance your reasoning process before providing an answer.\n1. **Self-Enhanced Reasoning:** Based on the initial Chain-of-Thought, reformulate it into an even clearer, more structured, and logically rigorous thought process.\n2. **Final Answer:** Based on your new, enhanced reasoning, provide the final clinical answer.\n\n**Question:** {ex['question']}\n**Initial Chain-of-Thought:**\n{ex['medcot_cot']}<end_of_turn>\n<start_of_turn>model\n**Self-Enhanced Reasoning:**\n{ex['medcot_cot']}\n\n**Final Answer:**\n{ex['answer']}<end_of_turn><eos>"
        }
    else:
        raise ValueError(f"Unknown prompt template type: {template_type}")

def prepare_dpo_dataset(dataset):
    """
    Chuẩn bị dataset cho DPO.
    - prompt: Câu hỏi
    - chosen: Câu trả lời đúng (dựa trên MedCOT)
    - rejected: Câu trả lời sai (dựa trên Default CoT)
    """
    def format_dpo(example):
        return {
            "prompt": f"Question: {example['question']}\nReasoning:",
            "chosen": f"{example['medcot_cot']}\nAnswer: {example['answer']}",
            "rejected": f"{example['default_cot']}\nAnswer: {example['answer']}" # Giả định câu trả lời giống nhau, chỉ CoT khác nhau
        }
    return dataset.map(format_dpo)


def main(config_path: str):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"🚀 Starting run: {config['run_name']} with mode: {config['train_mode'].upper()}")

    # 2. Load Dataset
    dataset = load_dataset("json", data_files=config['dataset_path'], split="train")

    # 3. Load Model & Tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config['base_model'], quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(**config['peft_config'], task_type="CAUSAL_LM") if config.get('peft_config') else None

    # 4. Initialize Trainer based on mode
    trainer = None
    training_args = SFTConfig(output_dir=f"models/checkpoints/{config['run_name']}", **config['training_args'])

    if config['train_mode'] == 'sft':
        formatter = get_prompt_formatter(config['prompt_template']['type'])
        formatted_dataset = dataset.map(formatter)
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=formatted_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            tokenizer=tokenizer,
        )
    elif config['train_mode'] == 'dpo':
        dpo_dataset = prepare_dpo_dataset(dataset)
        
        trainer = DPOTrainer(
            model,
            ref_model=None, # TRL sẽ tự tạo ref_model
            args=training_args,
            beta=config['dpo_config']['beta'],
            train_dataset=dpo_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    else:
        raise ValueError(f"Invalid train_mode: {config['train_mode']}")

    # 5. Train
    print("🔥 Training is starting...")
    trainer.train()
    trainer.save_model(config['output_dir'])
    print(f"✅ Training Done! Model saved to: {config['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the model config YAML file.")
    args = parser.parse_args()
    main(args.config)