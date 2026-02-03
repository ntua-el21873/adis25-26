import re
import torch
from typing import Tuple, List, Set, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

MODEL_ID = "openai-community/gpt2-xl"

# Lightweight keyword set for identifier gating (not a full SQL parser).
SQL_KEYWORDS: Set[str] = {
    "select", "from", "where", "join", "left", "right", "inner", "outer", "cross",
    "on", "as", "and", "or", "not", "in", "is", "null", "like", "distinct",
    "group", "by", "order", "having", "limit", "offset", "asc", "desc",
    "count", "sum", "avg", "min", "max", "between", "exists", "union", "all",
    "case", "when", "then", "else", "end"
}


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
        print(f"⏳ Loading {MODEL_ID} locally... (this might take a minute)")
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
        # Keep SELECT anchor, but include a lightweight format hint (no dataset-specific example).
        self._suffix_template = "\n\n### SQL:\nSELECT "

        # Pre-tokenize constant segments
        self._prefix_ids = self.tokenizer(self._prefix, add_special_tokens=False).input_ids
        self._rules_ids = self.tokenizer(self._rules, add_special_tokens=False).input_ids
        self._mid_ids = self.tokenizer(self._mid, add_special_tokens=False).input_ids
        self._suffix_ids = self.tokenizer(self._suffix_template, add_special_tokens=False).input_ids

        # Bad phrases to reduce prompt-channel drift and known degeneracy ("ids").
        self._bad_phrases = [
            "###",
            "Database schema",
            "Question",
            "Rules",
            "ids",
            "ids.",
        ]
        self._bad_words_ids = [
            self.tokenizer(p, add_special_tokens=False).input_ids
            for p in self._bad_phrases
            if len(self.tokenizer(p, add_special_tokens=False).input_ids) > 0
        ]

        # Force " FROM " somewhere in the output to reduce SELECT-lists without FROM.
        self._force_from_ids = self.tokenizer(" FROM ", add_special_tokens=False).input_ids

        # Markers that indicate template/garbage outputs (reject candidates that contain them).
        self._reject_markers = [
            "????",
            "<columns>",
            "<table>",
            "<condition>",
            "----------------",
        ]

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
    def build_prompt(self, schema: str, question: str) -> str:
        tables_view = self._schema_to_tables_bullets(schema)
        schema_block = tables_view if tables_view else schema

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

        # Use tables bullets view for copyability; truncate by lines.
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
            keep = max(
                32,
                budget - (len(self._prefix_ids) + len(self._rules_ids) + len(self._mid_ids) + len(self._suffix_ids))
            )
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

    # ----------------------------
    # SQL extraction / sanitation / gating
    # ----------------------------
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

    @staticmethod
    def _sanitize_sql_text(s: str) -> str:
        # Replace NBSP and other weird spaces with normal spaces
        s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
        # Normalize newlines
        s = s.replace("\r\n", "\n")
        return s

    @staticmethod
    def _strip_string_literals(s: str) -> str:
        # Replace quoted strings to avoid capturing identifiers inside them.
        s = re.sub(r"'([^'\\]|\\.)*'", "''", s)
        s = re.sub(r'"([^"\\]|\\.)*"', '""', s)
        return s

    def _extract_identifiers(self, sql: str) -> Tuple[set[str], set[str]]:
        """
        Lightweight identifier extraction for gating (NOT a SQL parser).
        Returns (tables_used, cols_used) in lowercase.

        - tables: tokens after FROM/JOIN
        - aliases: collected from FROM/JOIN ... [AS] alias
        - cols: dotted references + bare identifiers (excluding SQL keywords/tables/aliases)
        """
        s = sql.lower()
        s = self._strip_string_literals(s)

        # tables after FROM/JOIN
        tables = set(re.findall(r"\bfrom\s+([a-z_][a-z0-9_]*)\b", s))
        tables |= set(re.findall(r"\bjoin\s+([a-z_][a-z0-9_]*)\b", s))

        # aliases after FROM/JOIN
        alias_pairs = re.findall(r"\bfrom\s+([a-z_][a-z0-9_]*)\s+(?:as\s+)?([a-z_][a-z0-9_]*)\b", s)
        alias_pairs += re.findall(r"\bjoin\s+([a-z_][a-z0-9_]*)\s+(?:as\s+)?([a-z_][a-z0-9_]*)\b", s)
        aliases = {alias for (_, alias) in alias_pairs}

        # dotted references: t.col / a.col
        dotted = re.findall(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", s)
        cols = {c for (_, c) in dotted}

        # bare identifiers: any word-like token that's not a keyword/table/alias
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
    def generate_sql(self, schema: str, question: str, max_new_tokens: int = 128, max_time: float | None = None) -> str:
        inputs = self._make_inputs_under_limit(schema, question, max_new_tokens=max_new_tokens)
        input_len = inputs.pop("input_len")

        stopping = StoppingCriteriaList([_SQLStoppingCriteria(self.tokenizer, input_len)])

        # schema whitelist (use original compact schema, not bullet view)
        allowed_tables, allowed_cols = self._parse_schema_identifiers(schema)

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

                bad_words_ids=self._bad_words_ids if self._bad_words_ids else None,
                force_words_ids=[self._force_from_ids],
                return_dict_in_generate=True,
                output_scores=True,

                max_time=max_time,
            )

        seqs = out.sequences
        seq_scores = getattr(out, "sequences_scores", None)

        # Keep raw candidates for fallback selection
        raw_candidates: List[Tuple[float, str]] = []
        raw_with_from: List[Tuple[float, str]] = []

        # Valid candidates after gating
        candidates: List[Tuple[float, str]] = []

        for i in range(seqs.size(0)):
            gen_ids = seqs[i][input_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_text = self._sanitize_sql_text(gen_text)

            candidate = "SELECT " + gen_text
            candidate = self._sanitize_sql_text(candidate)
            sql = self._extract_sql(candidate).strip()

            score = float(seq_scores[i].item()) if seq_scores is not None else 0.0

            raw_candidates.append((score, sql))
            if " from " in f" {sql.lower()} ":
                raw_with_from.append((score, sql))

            # Reject obvious template/garbage placeholders
            if any(m in sql for m in self._reject_markers):
                continue

            # Must contain FROM (these datasets are overwhelmingly SELECT-FROM-WHERE)
            if " from " not in f" {sql.lower()} ":
                continue

            # Identifier gate (tables + columns, including bare columns)
            used_tables, used_cols = self._extract_identifiers(sql)

            if used_tables and not used_tables.issubset(allowed_tables):
                continue

            if used_cols and not used_cols.issubset(allowed_cols):
                continue

            candidates.append((score, sql))

        # If nothing passes, fallback to best candidate that at least contains FROM; else best raw.
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

        # pick best score among valid candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_sql = candidates[0][1].strip()

        if self.debug:
            print("=== CANDIDATES (valid) ===")
            for sc, s in candidates:
                print(sc, s)

        return best_sql
