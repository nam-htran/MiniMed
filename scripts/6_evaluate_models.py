import yaml
import argparse
import pandas as pd
import json
import torch
import gc
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from openai import OpenAI
from pathlib import Path

# Import Retriever cho RAG
from src.utils.faiss_search import faiss_retriever

# Lấy API Key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

# ==============================================================================
# 1. HÀM TẢI DỮ LIỆU
# ==============================================================================
def load_test_data(file_path, limit=None):
    print(f"📖 Reading {file_path}...")
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    if limit:
        return records[:limit]
    return records

# ==============================================================================
# 2. CÁC ENGINE INFERENCE (API, RAG, LOCAL)
# ==============================================================================

def run_api_inference(model_cfg, test_data):
    """Chạy model qua API (OpenAI hoặc Groq)."""
    print(f"\n☁️  Running API Model: {model_cfg['id']}")
    
    # Cấu hình Client
    if model_cfg['client'] == 'openai':
        client = OpenAI(api_key=OPENAI_KEY)
    elif model_cfg['client'] == 'groq':
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)
    else:
        return ["Config Error: Unknown Client"] * len(test_data)

    outputs = []
    for item in tqdm(test_data, desc=f"Evaluating {model_cfg['id']}"):
        # Prompt PubMedQA: Context + Question -> Answer
        prompt = f"""
        Context: {item.get('Context', '')}
        Question: {item['Question']}
        
        Answer with 'yes', 'no', or 'maybe' followed by a brief explanation.
        Answer:
        """
        try:
            resp = client.chat.completions.create(
                model=model_cfg['model_name'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256
            )
            outputs.append(resp.choices[0].message.content.strip())
        except Exception as e:
            outputs.append(f"API Error: {e}")
            time.sleep(1) # Tránh rate limit
            
    return outputs

def run_rag_inference(model_cfg, test_data):
    """Chạy Standard RAG: Retrieve Graph Nodes -> GPT-4o Answer."""
    print(f"\n📚  Running RAG Model: {model_cfg['id']}")
    
    if faiss_retriever is None:
        return ["FAISS Error: Retriever not loaded"] * len(test_data)
        
    client = OpenAI(api_key=OPENAI_KEY)
    outputs = []
    
    for item in tqdm(test_data, desc=f"Evaluating {model_cfg['id']}"):
        try:
            # 1. Retrieve Knowledge from Graph
            docs = faiss_retriever.search(item['Question'], k=model_cfg.get('retriever_top_k', 3))
            
            # Format retrieved knowledge
            kg_context = "\n".join([f"- {d['name']} ({d['labels'][0]})" for d in docs])
            
            # 2. Generate Answer (Hybrid Context: Abstract + Graph)
            prompt = f"""
            Abstract Context: {item.get('Context', '')}
            
            Knowledge Graph Info:
            {kg_context}
            
            Question: {item['Question']}
            
            Based on the Abstract and Knowledge Graph, answer with 'yes', 'no', or 'maybe'.
            Answer:
            """
            
            resp = client.chat.completions.create(
                model=model_cfg['llm_config']['model_name'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            outputs.append(resp.choices[0].message.content.strip())
            
        except Exception as e:
            outputs.append(f"RAG Error: {e}")
            
    return outputs

def run_local_inference(model_cfg, test_data):
    """Chạy Local Adapter (MedCOT SFT/DPO, Default DPO)."""
    adapter_path = model_cfg['adapter_path']
    base_model_id = model_cfg['base_model']
    print(f"\n🖥️  Running Local Model: {model_cfg['id']} (Adapter: {adapter_path})")

    if not os.path.exists(adapter_path):
        print(f"   ⚠️ Adapter path not found: {adapter_path}")
        return ["Adapter Missing"] * len(test_data)

    # Load Model (4-bit quantization để tiết kiệm VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    try:
        print("   ⏳ Loading Base Model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            quantization_config=bnb_config, 
            device_map="auto"
        )
        print(f"   ⏳ Loading Adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model.eval()
    except Exception as e:
        print(f"   ❌ Load Error: {e}")
        return [f"Load Error: {e}"] * len(test_data)

    outputs = []
    print("   🚀 Generating...")
    for item in tqdm(test_data, desc=f"Evaluating {model_cfg['id']}"):
        # Prompt đơn giản cho mô hình local
        prompt = f"Context: {item.get('Context', '')}\nQuestion: {item['Question']}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # Lấy phần mới sinh ra sau prompt
        answer = full_text[len(prompt):].strip()
        outputs.append(answer)

    # Dọn dẹp VRAM sau khi chạy xong model này
    del model, base_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return outputs

# ==============================================================================
# 3. HÀM CHẤM ĐIỂM (JUDGE)
# ==============================================================================
def evaluate_pubmedqa(item, generated_answer):
    """
    Sử dụng GPT-4o để đánh giá xem câu trả lời có khớp với nhãn (yes/no/maybe) không.
    """
    client = OpenAI(api_key=OPENAI_KEY)
    correct_label = item['Correct Answer'] # yes/no/maybe
    
    # Prompt cho Thẩm phán (Judge)
    judge_prompt = f"""
    You are a biomedical expert evaluator.
    
    Question: {item['Question']}
    Ground Truth Label: {correct_label}
    
    Model Prediction: "{generated_answer}"
    
    Task: 
    1. Determine if the Model Prediction implies 'yes', 'no', or 'maybe'.
    2. Check if it matches the Ground Truth Label.
    
    Return ONLY a JSON object: {{"predicted_label": "yes/no/maybe/unknown", "match": true/false}}
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        res = json.loads(resp.choices[0].message.content)
        return res.get("match", False), res.get("predicted_label", "unknown")
    except Exception:
        return False, "error"

# ==============================================================================
# 4. HÀM MAIN
# ==============================================================================
def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Tải Dữ liệu
    test_data = load_test_data(config['test_file'], config['limit'])
    results_df = pd.DataFrame(test_data)
    
    # 2. Chạy Inference cho từng Model
    for model_cfg in config['models_to_evaluate']:
        m_type = model_cfg['type']
        
        if m_type == 'api':
            preds = run_api_inference(model_cfg, test_data)
        elif m_type == 'rag':
            preds = run_rag_inference(model_cfg, test_data)
        elif m_type == 'local':
            preds = run_local_inference(model_cfg, test_data)
        else:
            preds = ["Unknown Type"] * len(test_data)
            
        results_df[f"{model_cfg['id']}_output"] = preds

    # 3. Chấm điểm (Evaluation)
    print("\n👨‍⚖️  Judging Results with GPT-4o...")
    scores = {}
    
    for model_cfg in config['models_to_evaluate']:
        mid = model_cfg['id']
        corrects = []
        labels = []
        
        # Duyệt qua kết quả để chấm
        for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc=f"Judging {mid}"):
            is_match, lbl = evaluate_pubmedqa(row.to_dict(), row[f"{mid}_output"])
            corrects.append(is_match)
            labels.append(lbl)
            
        results_df[f"{mid}_correct"] = corrects
        results_df[f"{mid}_pred_label"] = labels
        
        # Tính Accuracy
        acc = sum(corrects) / len(corrects) * 100
        scores[mid] = acc
        print(f"   -> {mid} Accuracy: {acc:.2f}%")

    # 4. Lưu kết quả
    results_df.to_csv(config['output_csv'], index=False)
    print(f"\n✅ Evaluation Finished! Results saved to: {config['output_csv']}")
    print("🏆 FINAL LEADERBOARD:")
    for m, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {m}: {s:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/evaluate_config.yaml")
    args = parser.parse_args()
    main(args.config)