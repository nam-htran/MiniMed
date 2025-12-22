import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import gc

logger = logging.getLogger("LOCAL_LLM")

# Model 1.5B tối ưu cho 3050 Ti
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

class LocalCoTGenerator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalCoTGenerator, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
        return cls._instance

    def load_model(self):
        """Load model với cấu hình tối ưu VRAM"""
        if self.model is not None:
            return

        logger.info(f"⏳ Loading Local CoT Model trên RTX 3050 Ti: {MODEL_ID}...")
        try:
            # Cấu hình 4-bit
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )

            # --- SỬA LẠI: BỎ force_download=True NẾU ĐÃ TẢI XONG ---
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            
            # --- FIX CẢNH BÁO PAD TOKEN ---
            # Qwen/DeepSeek thường không có pad token mặc định, ta gán nó bằng EOS token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info(f"✅ Model đã load! VRAM dự kiến: ~1.5GB")
            
        except Exception as e:
            logger.critical(f"❌ Lỗi load model: {e}")
            raise e

    def generate_cot(self, prompt: str) -> str:
        if self.model is None:
            self.load_model()

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize inputs
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)

        # --- FIX CẢNH BÁO ATTENTION MASK ---
        # Tạo mask: 1 cho token thật, 0 cho padding (ở đây toàn bộ là thật vì batch=1)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,    # Truyền mask vào
                pad_token_id=self.tokenizer.pad_token_id, # Truyền pad_token_id
                max_new_tokens=2048,
                temperature=0.6,
                do_sample=True,
                top_p=0.9
            )

        response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return response.strip()

    def unload(self):
        """Giải phóng VRAM"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("🗑️ Đã giải phóng Model khỏi GPU")

# Singleton
local_llm = LocalCoTGenerator()