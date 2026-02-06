<div align="center">

# Local LLM Text-to-SQL Benchmark  
### GPT-2 XL vs Qwen2.5-Coder · MySQL & MariaDB

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.45+-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-MySQL%20%7C%20MariaDB-2496ED.svg)
![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)

**Execution-based benchmarking of fully local LLM Text-to-SQL systems**  
*No APIs · No cloud · Reproducible · Dialect-aware*

</div>

This repository contains an **end-to-end, fully local Text-to-SQL benchmarking framework** that evaluates open-source Large Language Models (LLMs) by **execution accuracy** (semantic correctness) rather than string matching.  

The system integrates local LLM agents with **Dockerized MySQL 8.0** and **MariaDB 11.2**, executes generated queries across multiple real-world datasets, and logs detailed, execution-level results for reproducible analysis.

## Key Features

- **Fully local inference** (no external APIs)
- **Execution-based evaluation**: compares result sets from predicted vs gold SQL
- **Two model agents**:
  - **GPT-2 XL** (generalist, 1024-token context)
  - **Qwen2.5-Coder-1.5B-Instruct** (code-specialized, larger context)
- **Cross-dialect benchmarking**: MySQL vs MariaDB
- **Deterministic runs**:
  - fixed decoding configuration (beam search, no sampling)
  - deterministic schema compaction and truncation
- **Schema compaction** with deterministic table ranking + optional FK-based 1-hop expansion
- **Robust logging** in JSONL + post-processing scripts to generate aggregated CSVs and plots

---

## Repository Structure

```

.
├── models/                     # Model agents (GPT-2 XL, Qwen)
├── database/                   # DB connectivity + DatabaseManager
├── scripts/
│   ├── setup/                  # Download/bootstrap/test scripts
│   ├── llm/                    # Smoketests + single-query runners
│   ├── plots/                  # Aggregation + plotting utilities
|   ├── run_gpt2xl_benchmark.py # GPT2-XL benchmark
|   ├── run_qwen_benchmark.py   # Qwen-2.5 benchmark
│   └── sql_utils.py            # Normalization/repairs/comparison utilities
├── data/
│   └── processed/              # Generated DDL/DML dumps (optional)
├── datasets_source/
│   └── data/                   # Dataset JSON files (advising/atis/imdb/yelp)
├── source_sql/                 # Manually added SQL assets (IMDb/Yelp)
├── results/                    # JSONL logs and aggregated outputs
├── docker-compose.yml          # MySQL/MariaDB containers + services
├── requirements.txt            # Pinned Python dependencies
└── setup.ps1                   # Windows end-to-end setup script

````

---

## Requirements

### Software
- **Python 3.12** (recommended; pinned dependencies are tested with this)
- **Docker + Docker Compose**
- (Optional) **CUDA** for GPU inference

### Python dependencies
Pinned in `requirements.txt` (includes `transformers`, `torch`, `SQLAlchemy`, `PyMySQL`, `pandas`, etc.).

---

## Quick Start (Windows)

The easiest way to set up everything is:

1. Open PowerShell in the repository root
2. Run:
```powershell
.\setup.ps1
````

This script:

* validates prerequisites (Python/Docker)
* creates a virtual environment
* installs dependencies
* creates `.env` (if missing)
* starts DB containers
* bootstraps datasets
* validates DB connections

---

## Running Benchmarks

### Qwen baseline

```bash
python scripts/run_qwen_baseline.py --dataset datasets_source/data/{dataset}.json --rdbms {rdbms} --max_tables {your_choice}
```

### GPT-2 XL baseline

```bash
python scripts/run_gpt2xl_baseline.py --dataset datasets_source/data/{dataset}.json --rdbms {rdbms} --max_tables {your_choice}
```

Notes:

* `--backend` typically accepts `mysql` or `mariadb` (or both)
* `--max_tables` activates schema compaction (deterministic relevance ranking)
* Outputs are written under `results/` as **JSONL** (one record per question)

---

## Smoke Tests (Recommended Before Full Runs)

### Qwen agent smoke test

```bash
python scripts/llm/qwen_smoketest.py --dataset datasets_source/data/{dataset}.json --rdbms {rdbms} --max_tables {your_choice}
```

### End-to-end Text-to-SQL smoke test

```bash
python scripts/llm/llm_smoketest_text2sql.py --dataset datasets_source/data/{dataset}.json --rdbms {rdbms} --max_tables {your_choice}
```

### Single-query debug runners

```bash
python scripts/llm/qwen_one_query_run.py
python scripts/llm/llm_one_query_run.py
```

These are useful to validate prompt formatting and decoding without running full datasets.

---

## Evaluation: What "Correct" Means Here

This framework evaluates models using **Execution Accuracy (EX)**:

* Executes **predicted SQL** and **gold SQL**
* Compares **result sets** rather than SQL strings
* Logs:

  * executability (did it run?)
  * correctness (did it match gold results?)
  * execution latency
  * generation latency
  * prompt token counts
  * repairs/normalization details

This avoids penalizing semantically correct queries that differ syntactically (aliases, formatting, equivalent joins, etc.).

---

## Results, Aggregation, and Plots

After producing JSONL logs in `results/`, run the post-processing scripts (names may vary slightly by version):

```bash
python scripts/plots/build_master_csv.py
python scripts/plots/make_*.py
```

Typical outputs:

* `master_results.csv` (merged analysis table)
* plots under `docs/figures/...`
* tables under `docs/tables/...`

---

## Configuration

Environment variables are stored in `.env` (auto-generated by `setup.ps1` if missing). Typical variables include:

* DB credentials/ports
* dataset paths (e.g., `DATASETS_SOURCE_PATH`)
* processed output paths

If you modify ports or container names, ensure the database connection utilities in `database/` match.

---

