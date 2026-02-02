import re
import torch
from typing import Tuple, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

MODEL_ID = "openai-community/gpt2-xl"


class _SQLStoppingCriteria(StoppingCriteria):
    """
    Stop when the generated continuation looks 'complete' or drifts into prompt headers.
    We only inspect the NEW tokens (continuation) decoded as text.
    """
    def __init__(self, tokenizer, input_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0][self.input_len:]
        if gen_ids.numel() == 0:
            return False

        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 1) Stop at semicolon (ideal SQL terminator)
        if ";" in text:
            return True

        # 2) Stop at blank line (often ends the SQL)
        if "\n\n" in text or "\r\n\r\n" in text:
            return True

        # 3) Stop if model starts emitting prompt headers again
        if "###" in text:
            return True

        return False


class GPT2XLAgent:
    def __init__(self, device: str | None = None, debug: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_ctx = getattr(self.model.config, "n_positions", 1024)

        # Prompt pieces
        self._prefix = "### Database schema:\n"
        self._rules = (
            "\n\n### Rules:\n"
            "- Use ONLY table and column names from the schema.\n"
            "- Do NOT invent table or column names.\n"
            "- Return exactly ONE SQL query.\n"
            "- End the query with a semicolon.\n"
        )
        self._mid = "\n\n### Question:\n"
        self._suffix_template = "\n\n### SQL:\n-- Format: SELECT <columns> FROM <table> WHERE <condition>;\nSELECT "


        # Pre-tokenize constant segments
        self._prefix_ids = self.tokenizer(self._prefix, add_special_tokens=False).input_ids
        self._rules_ids = self.tokenizer(self._rules, add_special_tokens=False).input_ids
        self._mid_ids = self.tokenizer(self._mid, add_special_tokens=False).input_ids
        self._suffix_ids = self.tokenizer(self._suffix_template, add_special_tokens=False).input_ids
        
        

        self._bad_phrases = [
            "###",
            "Database schema",
            "Question",
            "Rules",
            "ids",
            "ids."
        ]
        self._bad_words_ids = [
            self.tokenizer(p, add_special_tokens=False).input_ids
            for p in self._bad_phrases
            if len(self.tokenizer(p, add_special_tokens=False).input_ids) > 0
        ]

    def _parse_schema_identifiers(self, schema: str) -> Tuple[set[str], set[str]]:
        tables = set()
        cols = set()
        for ln in schema.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", ln)
            if not m:
                continue
            t = m.group(1).lower()
            tables.add(t)
            for c in [x.strip().lower() for x in m.group(2).split(",")]:
                if c:
                    cols.add(c)
        return tables, cols

    def _extract_identifiers_rough(self, sql: str) -> Tuple[set[str], set[str]]:
        """
        Rough identifier extraction:
        - tables from FROM/JOIN tokens
        - columns from patterns like t.col or bare col in SELECT/WHERE (very rough)
        This is meant for gating, not correctness.
        """
        s = sql.lower()
        table_hits = set(re.findall(r"\bfrom\s+([a-z_][a-z0-9_]*)\b", s))
        join_hits = set(re.findall(r"\bjoin\s+([a-z_][a-z0-9_]*)\b", s))
        tables = table_hits | join_hits

        # t.col patterns
        dotted_cols = set(re.findall(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", s))
        cols = {c for (_, c) in dotted_cols}

        # bare columns are too ambiguous; we avoid hard-gating on them
        return tables, cols

    @staticmethod
    def _schema_to_tables_bullets(schema: str) -> str:
        """
        Convert lines like:
          flights(id, origin, destination)
        into:
          Tables:
          - flights: id, origin, destination
        If parsing fails, return empty string (no harm).
        """
        lines = [ln.strip() for ln in schema.splitlines() if ln.strip()]
        rows = []
        for ln in lines:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", ln)
            if not m:
                continue
            t = m.group(1)
            cols = m.group(2).strip()
            rows.append(f"- {t}: {cols}")

        if not rows:
            return ""

        return "Tables:\n" + "\n".join(rows) + "\n"

    def build_prompt(self, schema: str, question: str) -> str:
        tables_view = self._schema_to_tables_bullets(schema)
        if tables_view:
            schema_block = tables_view
        else:
            schema_block = schema

        return (
            f"{self._prefix}{schema_block}"
            f"{self._rules}"
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
            # Tokenize line with trailing newline to match actual usage
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

        # Build schema block with redundant tables view, then truncate safely by lines.
        tables_view = self._schema_to_tables_bullets(schema)
        schema_full = tables_view if tables_view else schema

        question_ids = self.tokenizer(question, add_special_tokens=False).input_ids

        fixed_len = (
            len(self._prefix_ids)
            + len(self._rules_ids)
            + len(self._mid_ids)
            + len(question_ids)
            + len(self._suffix_ids)
        )

        # If question too long, truncate (rare)
        if fixed_len > budget:
            keep = max(32, budget - (len(self._prefix_ids) + len(self._rules_ids) + len(self._mid_ids) + len(self._suffix_ids)))
            question_ids = question_ids[-keep:]
            fixed_len = (
                len(self._prefix_ids)
                + len(self._rules_ids)
                + len(self._mid_ids)
                + len(question_ids)
                + len(self._suffix_ids)
            )

        schema_budget = max(0, budget - fixed_len)

        schema_trunc = self._truncate_schema_by_lines(schema_full, schema_budget)
        schema_ids = self.tokenizer(schema_trunc, add_special_tokens=False).input_ids

        input_ids = self._prefix_ids + schema_ids + self._rules_ids + self._mid_ids + question_ids + self._suffix_ids
        attn = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor([input_ids], device=self.device),
            "attention_mask": torch.tensor([attn], device=self.device),
            "input_len": len(input_ids),
        }

    def generate_sql(self, schema: str, question: str, max_new_tokens: int = 128) -> str:
        inputs = self._make_inputs_under_limit(schema, question, max_new_tokens=max_new_tokens)
        input_len = inputs.pop("input_len")

        stopping = StoppingCriteriaList([_SQLStoppingCriteria(self.tokenizer, input_len)])

        # schema whitelist
        allowed_tables, allowed_cols = self._parse_schema_identifiers(schema)

        force_from = self.tokenizer(" FROM ", add_special_tokens=False).input_ids

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,

                # deterministic beam search
                do_sample=False,
                num_beams=4,
                num_return_sequences=4,
                early_stopping=True,
                length_penalty=0.9,

                # stop when we see ';' / blank line / header drift
                stopping_criteria=stopping,

                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,

                bad_words_ids=self._bad_words_ids,
                force_words_ids=[force_from],
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        seqs = out.sequences
        seq_scores = out.sequences_scores

        candidates: List[Tuple[float, str]] = []
        for i in range(seqs.size(0)):
            gen_ids = seqs[i][input_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_text = self._sanitize_sql_text(gen_text)

            candidate = "SELECT " + gen_text
            candidate = self._sanitize_sql_text(candidate)
            sql = self._extract_sql(candidate).strip()

            # Basic must-have: include FROM (your tasks basically always need it)
            if " from " not in f" {sql.lower()} ":
                continue

            # Identifier gate
            used_tables, used_cols = self._extract_identifiers_rough(sql)
            if used_tables and not used_tables.issubset(allowed_tables):
                continue
            # dotted cols gate (safe-ish)
            if used_cols and not used_cols.issubset(allowed_cols):
                continue

            candidates.append((float(seq_scores[i].item()), sql))

        # If nothing passes, fall back to best raw (still cleaned)
        if not candidates:
            best_i = int(torch.argmax(seq_scores).item())
            gen_ids = seqs[best_i][input_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_text = self._sanitize_sql_text(gen_text)
            sql = self._extract_sql("SELECT " + gen_text).strip()
            return sql

        # pick best score among valid candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_sql = candidates[0][1]

        if self.debug:
            print("=== CANDIDATES (valid) ===")
            for sc, s in candidates:
                print(sc, s)            
        return best_sql




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
    
    def _sanitize_sql_text(self, s: str) -> str:
        # Replace NBSP and other weird spaces with normal spaces
        s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
        # Normalize newlines
        s = s.replace("\r\n", "\n")
        return s

