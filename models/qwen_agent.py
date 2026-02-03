import re
import torch
from typing import Tuple, List, Set, Dict, Any, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Lightweight keyword set for identifier gating (not a full SQL parser).
SQL_KEYWORDS: Set[str] = {
    "select", "from", "where", "join", "left", "right", "inner", "outer", "cross",
    "on", "as", "and", "or", "not", "in", "is", "null", "like", "distinct",
    "group", "by", "order", "having", "limit", "offset", "asc", "desc",
    "count", "sum", "avg", "min", "max", "between", "exists", "union", "all",
    "case", "when", "then", "else", "end"
}

# Hard "drift" markers: if the model starts emitting these, we stop immediately.
DRIFT_MARKERS: List[str] = [
    "\n###",  # prompt headers coming back
    "### ",   # sometimes no leading newline
    "Explanation:",
    "Rationale:",
    "Human:",
    "Assistant:",
    "I'm sorry",
    "I can't",
    "I cannot",
    "Write a SQL query",
    "Human resources",
    "policy that",
    "criteria.",
]


class _SQLStoppingCriteria(StoppingCriteria):
    """
    Stop when the generated continuation looks 'complete' or drifts into non-SQL content.
    We inspect ONLY the NEW tokens (continuation) decoded as text.
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

        # 2) Stop at blank line (often ends the SQL or begins explanations)
        if "\n\n" in text or "\r\n\r\n" in text:
            return True

        # 3) Stop on drift markers (chat artifacts / explanations / headers)
        for m in DRIFT_MARKERS:
            if m in text:
                return True

        return False


class QwenAgent:
    def __init__(self, debug: bool = False):
        print(f"⏳ Loading {MODEL_ID} locally... (this might take a minute)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug = debug

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # Qwen instruct models typically work best with chat templates.
        # Also: device_map should NOT be "cuda"/"cpu"; use "auto" or move the model manually.
        # We'll do the simplest reliable approach: load then .to(device).
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        self.model.eval()

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Context length (use best available config field)
        self.max_ctx = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.model.config, "n_positions", None)
            or 4096
        )

        # Prompt pieces (used if chat template is unavailable for some reason)
        self._prefix = "### Database schema:\n"
        self._rules = (
            "\n\n### Rules:\n"
            "- Use ONLY table and column names from the schema.\n"
            "- Do NOT invent table or column names.\n"
            "- Return exactly ONE SQL query.\n"
            "- Output ONLY SQL (no explanation).\n"
            "- End the query with a semicolon (\";\").\n"
        )
        self._mid = "\n\n### Question:\n"
        # Strong suffix anchor
        self._suffix_template = "\n\n### SQL:\n-- Output one SQL statement ending with ;\nSELECT "

        # Pre-tokenize constant segments
        self._prefix_ids = self.tokenizer(self._prefix, add_special_tokens=False).input_ids
        self._rules_ids = self.tokenizer(self._rules, add_special_tokens=False).input_ids
        self._mid_ids = self.tokenizer(self._mid, add_special_tokens=False).input_ids
        self._suffix_ids = self.tokenizer(self._suffix_template, add_special_tokens=False).input_ids

        # Bad phrases to reduce prompt-channel drift.
        self._bad_phrases = [
            "###",
            "Database schema",
            "Question",
            "Rules",
            "Explanation",
            "Human:",
            "Assistant:",
        ]
        self._bad_words_ids = [
            self.tokenizer(p, add_special_tokens=False).input_ids
            for p in self._bad_phrases
            if len(self.tokenizer(p, add_special_tokens=False).input_ids) > 0
        ]

        # Force " FROM " somewhere in output to reduce SELECT-lists without FROM.
        self._force_from_ids = self.tokenizer(" FROM ", add_special_tokens=False).input_ids

        # Markers that indicate template/garbage outputs.
        self._reject_markers = [
            "????",
            "<columns>",
            "<table>",
            "<condition>",
            "----------------",
        ]

        print(f"✅ Model loaded on {self.device.upper()}")

    # ----------------------------
    # Schema helpers
    # ----------------------------
    def _parse_schema_identifiers(self, schema: str) -> Tuple[set[str], set[str]]:
        """
        Parse compact schema lines:
          table(col1, col2, ...)
        Returns:
          allowed_tables (lowercase)
          allowed_cols (lowercase union)
        """
        tables: set[str] = set()
        cols: set[str] = set()
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

    @staticmethod
    def _schema_to_tables_bullets(schema: str) -> str:
        """
        Convert:
          movie(mid, title, release_year)
        into:
          Tables:
          - movie: mid, title, release_year
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

    # ----------------------------
    # Prompt building / truncation
    # ----------------------------
    def _build_chat_prompt(self, schema: str, question: str) -> str:
        """
        Preferred: use chat template for instruct model.
        Returns a text prompt ready for tokenization.
        """
        sys_msg = (
            "You are a SQL generator.\n"
            "Return exactly ONE SQL query.\n"
            "Use ONLY the provided schema tables/columns.\n"
            "Output ONLY SQL (no explanation, no markdown).\n"
            "End the SQL with a semicolon."
        )
        user_msg = (
            f"Database schema:\n{schema}\n\n"
            f"Question:\n{question}\n\n"
            f"Return the SQL now."
        )

        # Use chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Fallback to old-style prompt
        return (
            f"{self._prefix}{schema}"
            f"{self._rules}"
            f"{self._mid}{question}"
            f"{self._suffix_template}"
        )

    def _truncate_schema_by_lines(self, schema: str, schema_budget_tokens: int) -> str:
        """
        Keep whole lines (tables) until token budget is met.
        Prevents cutting identifiers mid-token and preserves structure.
        """
        if schema_budget_tokens <= 0:
            return ""

        lines = [ln for ln in schema.splitlines() if ln.strip()]
        kept = []
        used = 0
        for ln in lines:
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

        # For chat template, we need to rebuild prompt after truncation.
        # We'll truncate schema by lines using token budget estimated from fallback segments.
        tables_view = self._schema_to_tables_bullets(schema)
        schema_full = tables_view if tables_view else schema

        # Tokenize question alone (for rough budgeting)
        question_ids = self.tokenizer(question, add_special_tokens=False).input_ids

        # Rough fixed len (fallback prompt pieces) – good enough to decide schema truncation.
        fixed_len = (
            len(self._prefix_ids)
            + len(self._rules_ids)
            + len(self._mid_ids)
            + len(question_ids)
            + len(self._suffix_ids)
        )

        if fixed_len > budget:
            keep = max(
                32,
                budget - (len(self._prefix_ids) + len(self._rules_ids) + len(self._mid_ids) + len(self._suffix_ids))
            )
            question_ids = question_ids[-keep:]
            question = self.tokenizer.decode(question_ids, skip_special_tokens=True)

            fixed_len = (
                len(self._prefix_ids)
                + len(self._rules_ids)
                + len(self._mid_ids)
                + len(question_ids)
                + len(self._suffix_ids)
            )

        schema_budget = max(0, budget - fixed_len)
        schema_trunc = self._truncate_schema_by_lines(schema_full, schema_budget)

        prompt = self._build_chat_prompt(schema_trunc, question)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return {
            **inputs,
            "input_len": int(inputs["input_ids"].shape[1]),
            "prompt_text": prompt,
        }

    # ----------------------------
    # SQL extraction / sanitation / gating
    # ----------------------------
    @staticmethod
    def _sanitize_sql_text(s: str) -> str:
        s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
        s = s.replace("\r\n", "\n")
        return s

    @staticmethod
    def _strip_string_literals(s: str) -> str:
        s = re.sub(r"'([^'\\]|\\.)*'", "''", s)
        s = re.sub(r'"([^"\\]|\\.)*"', '""', s)
        return s

    @staticmethod
    def _cut_on_drift_markers(s: str) -> str:
        """
        If generation appended non-SQL content on the same line (no semicolon/newline),
        cut at the earliest drift marker.
        """
        cut_points = []
        for m in DRIFT_MARKERS:
            idx = s.find(m)
            if idx != -1:
                cut_points.append(idx)
        if cut_points:
            return s[: min(cut_points)].strip()
        return s.strip()

    @classmethod
    def _extract_sql(cls, text: str) -> str:
        """
        Extract a single SQL statement from text.
        Priority:
          1) first semicolon
          2) blank line / newline
          3) drift markers (Explanation/Human/etc.)
        """
        text = text.strip()

        semi = text.find(";")
        if semi != -1:
            return text[: semi + 1].strip()

        for sep in ["\n\n", "\r\n\r\n", "\n", "\r\n"]:
            idx = text.find(sep)
            if idx != -1:
                return text[:idx].strip()

        # No delimiters: cut on drift markers if present
        return cls._cut_on_drift_markers(text)

    def _extract_identifiers(self, sql: str) -> Tuple[set[str], set[str]]:
        """
        Lightweight identifier extraction for gating (NOT a SQL parser).
        Returns (tables_used, cols_used) in lowercase.
        """
        s = sql.lower()
        s = self._strip_string_literals(s)

        tables = set(re.findall(r"\bfrom\s+([a-z_][a-z0-9_]*)\b", s))
        tables |= set(re.findall(r"\bjoin\s+([a-z_][a-z0-9_]*)\b", s))

        alias_pairs = re.findall(r"\bfrom\s+([a-z_][a-z0-9_]*)\s+(?:as\s+)?([a-z_][a-z0-9_]*)\b", s)
        alias_pairs += re.findall(r"\bjoin\s+([a-z_][a-z0-9_]*)\s+(?:as\s+)?([a-z_][a-z0-9_]*)\b", s)
        aliases = {alias for (_, alias) in alias_pairs}

        dotted = re.findall(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", s)
        cols = {c for (_, c) in dotted}

        tokens = re.findall(r"\b[a-z_][a-z0-9_]*\b", s)
        for tok in tokens:
            if tok in SQL_KEYWORDS:
                continue
            if tok in tables:
                continue
            if tok in aliases:
                continue
            if tok in {"true", "false"}:
                continue
            cols.add(tok)

        return tables, cols

    def _ok(self, sql: str) -> bool:
        return not any(m in sql for m in self._reject_markers)

    # ----------------------------
    # Main generation
    # ----------------------------
    def generate_sql(
        self,
        schema: str,
        question: str,
        max_new_tokens: int = 128,
        max_time: Optional[float] = None,
    ) -> str:
        pack = self._make_inputs_under_limit(schema, question, max_new_tokens=max_new_tokens)
        input_len = pack.pop("input_len")
        prompt_text = pack.pop("prompt_text", "")

        stopping = StoppingCriteriaList([_SQLStoppingCriteria(self.tokenizer, input_len)])

        allowed_tables, allowed_cols = self._parse_schema_identifiers(schema)

        with torch.no_grad():
            out = self.model.generate(
                **pack,
                max_new_tokens=max_new_tokens,

                # deterministic beam search
                do_sample=False,
                num_beams=4,
                num_return_sequences=4,
                early_stopping=True,
                length_penalty=0.9,

                temperature=None,
                top_p=None,
                top_k=None,

                stopping_criteria=stopping,

                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,

                bad_words_ids=self._bad_words_ids if self._bad_words_ids else None,
                force_words_ids=[self._force_from_ids],

                return_dict_in_generate=True,
                output_scores=True,

                max_time=max_time,
            )

        seqs = out.sequences
        seq_scores = getattr(out, "sequences_scores", None)

        raw_candidates: List[Tuple[float, str]] = []
        raw_with_from: List[Tuple[float, str]] = []
        candidates: List[Tuple[float, str]] = []

        for i in range(seqs.size(0)):
            gen_ids = seqs[i][input_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_text = self._sanitize_sql_text(gen_text)

            # Some chat templates may re-emit parts; keep extraction robust.
            # Build candidate anchored at SELECT to reduce leading noise.
            # 1) Strip fenced code blocks if present
            gen_text_clean = gen_text
            m = re.search(r"```(?:sql)?\s*(.*?)\s*```", gen_text_clean, re.DOTALL | re.IGNORECASE)
            if m:
                gen_text_clean = m.group(1).strip()

            gen_text_clean = self._sanitize_sql_text(gen_text_clean).strip()

            # 2) Extract one SQL statement (handles semicolon/newline/drift)
            sql = self._extract_sql(gen_text_clean).strip()

            # 3) Anchor only if needed (avoid SELECT SELECT)
            if not re.match(r"^\s*select\b", sql, flags=re.IGNORECASE):
                sql = "SELECT " + sql

            sql = sql.strip()

            # Ensure single trailing semicolon if it exists somewhere
            if ";" in sql:
                sql = sql.split(";", 1)[0].strip() + ";"


            # final “same-line drift” safety net
            sql = self._cut_on_drift_markers(sql)

            # If the prompt leaked into decoded text (rare), drop it.
            if prompt_text and sql.startswith(prompt_text):
                sql = sql[len(prompt_text):].strip()

            score = float(seq_scores[i].item()) if seq_scores is not None else 0.0

            raw_candidates.append((score, sql))
            if " from " in f" {sql.lower()} ":
                raw_with_from.append((score, sql))

            if any(m in sql for m in self._reject_markers):
                continue

            if " from " not in f" {sql.lower()} ":
                continue

            used_tables, used_cols = self._extract_identifiers(sql)

            if used_tables and not used_tables.issubset(allowed_tables):
                continue

            if used_cols and not used_cols.issubset(allowed_cols):
                continue

            candidates.append((score, sql))

        if not candidates:
            raw_with_from_ok = [(sc, s) for sc, s in raw_with_from if self._ok(s)]
            if raw_with_from_ok:
                raw_with_from_ok.sort(key=lambda x: x[0], reverse=True)
                chosen = raw_with_from_ok[0][1]
            else:
                raw_ok = [(sc, s) for sc, s in raw_candidates if self._ok(s)]
                if raw_ok:
                    raw_ok.sort(key=lambda x: x[0], reverse=True)
                    chosen = raw_ok[0][1]
                else:
                    raw_candidates.sort(key=lambda x: x[0], reverse=True)
                    chosen = raw_candidates[0][1] if raw_candidates else ""
            return chosen.strip()

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_sql = candidates[0][1].strip()

        if self.debug:
            print("=== CANDIDATES (valid) ===")
            for sc, s in candidates:
                print(sc, s)

        return best_sql
