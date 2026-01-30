import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

class QwenAgent:
    def __init__(self):
        print(f"⏳ Loading {MODEL_ID} locally... (this might take a minute)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32, 
            device_map=self.device
        )
        self.model.eval()
        print(f"✅ Model loaded on {self.device.upper()}")

    # Επιστρέφει tuple: (sql, prompt_tokens, completion_tokens)
    def generate_sql(self, schema: str, question: str, max_new_tokens: int = 256) -> tuple[str, int, int]:
        prompt = (
            f"### Database schema:\n{schema}\n\n"
            f"### Question:\n{question}\n\n"
            f"### SQL:\n"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 1. Υπολογισμός Prompt Tokens
        prompt_tokens = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        # 2. Υπολογισμός Completion Tokens (Generated only)
        # Αφαιρούμε το μήκος του prompt από το συνολικό μήκος
        completion_tokens = generated_ids.shape[1] - prompt_tokens
            
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # --- CLEANING ---
        if "### SQL:" in output_text:
            raw_answer = output_text.split("### SQL:")[-1].strip()
        else:
            raw_answer = output_text.replace(prompt, "").strip()

        code_block_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", raw_answer, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            sql = code_block_match.group(1).strip()
        else:
            sql = raw_answer

        if ";" in sql:
            sql = sql.split(";")[0] + ";"
            
        sql = sql.replace("```", "").strip()
        
        # Επιστροφή 3 τιμών
        return sql, prompt_tokens, completion_tokens