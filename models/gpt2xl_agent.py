import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "openai-community/gpt2-xl"

class GPT2XLAgent:
    def __init__(self, device: str | None = None, debug: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_ctx = getattr(self.model.config, "n_positions", 1024)

        # Pre-tokenize constant segments for speed/reproducibility
        self._prefix = "### Database schema:\n"
        self._mid = "\n\n### Question:\n"
        self._suffix_template = "\n\n### SQL:\nSELECT "  # NOTE trailing space

        self._prefix_ids = self.tokenizer(self._prefix, add_special_tokens=False).input_ids
        self._mid_ids = self.tokenizer(self._mid, add_special_tokens=False).input_ids
        self._suffix_ids = self.tokenizer(self._suffix_template, add_special_tokens=False).input_ids

    def build_prompt(self, schema: str, question: str) -> str:
        return (
            f"{self._prefix}{schema}"
            f"{self._mid}{question}"
            f"{self._suffix_template}"
        )

    def _truncate_schema_by_lines(self, schema: str, schema_budget_tokens: int) -> str:
        """
        Keep whole lines (tables) until token budget is met.
        This prevents cutting identifiers mid-token and preserves structure.
        """
        if schema_budget_tokens <= 0:
            return ""

        lines = [ln for ln in schema.splitlines() if ln.strip()]
        kept = []
        used = 0
        for ln in lines:
            # +1 for newline join effect
            ln_ids = self.tokenizer(ln + "\n", add_special_tokens=False).input_ids
            if used + len(ln_ids) > schema_budget_tokens:
                break
            kept.append(ln)
            used += len(ln_ids)

        return "\n".join(kept)

    def _make_inputs_under_limit(self, schema: str, question: str, max_new_tokens: int):
        budget = self.max_ctx - max_new_tokens
        if budget <= 0:
            raise ValueError(f"max_new_tokens={max_new_tokens} leaves no room for prompt in ctx={self.max_ctx}")

        question_ids = self.tokenizer(question, add_special_tokens=False).input_ids

        fixed_len = len(self._prefix_ids) + len(self._mid_ids) + len(question_ids) + len(self._suffix_ids)

        # If question too long, truncate question (rare)
        if fixed_len > budget:
            keep = max(32, budget - (len(self._prefix_ids) + len(self._mid_ids) + len(self._suffix_ids)))
            question_ids = question_ids[-keep:]
            fixed_len = len(self._prefix_ids) + len(self._mid_ids) + len(question_ids) + len(self._suffix_ids)

        schema_budget = max(0, budget - fixed_len)

        # Truncate schema safely by lines
        schema_trunc = self._truncate_schema_by_lines(schema, schema_budget)
        schema_ids = self.tokenizer(schema_trunc, add_special_tokens=False).input_ids

        input_ids = self._prefix_ids + schema_ids + self._mid_ids + question_ids + self._suffix_ids
        attn = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor([input_ids], device=self.device),
            "attention_mask": torch.tensor([attn], device=self.device),
            "input_len": len(input_ids),  # for slicing generated part
        }

    def generate_sql(self, schema: str, question: str, max_new_tokens: int = 64) -> str:
        inputs = self._make_inputs_under_limit(schema, question, max_new_tokens=max_new_tokens)
        input_len = inputs.pop("input_len")

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # reduces degeneracy a bit
            )

        # Decode ONLY generated continuation
        gen_ids = out[0][input_len:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Since prompt already ends with "SELECT ", reconstruct full SQL candidate
        candidate = "SELECT " + gen_text

        sql = self._extract_sql(candidate)

        if self.debug:
            print("=== CANDIDATE ===")
            print(candidate)
            print("=== EXTRACTED ===")
            print(sql)

        return sql.strip()

    @staticmethod
    def _extract_sql(text: str) -> str:
        # Stop at first semicolon if present, else first blank line or line break
        semi = text.find(";")
        if semi != -1:
            return text[: semi + 1].strip()

        for sep in ["\n\n", "\r\n\r\n", "\n", "\r\n"]:
            idx = text.find(sep)
            if idx != -1:
                return text[:idx].strip()

        return text.strip()
