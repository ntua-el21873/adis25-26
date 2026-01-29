"""
scripts/download_datasets.py
Download and analyze text2sql datasets (robust + reproducible)
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATASETS: Dict[str, Dict[str, str]] = {
    "advising": {
        "url": "https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/master/data/advising.json",
        "description": "205 queries, 15 tables",
    },
    "atis": {
        "url": "https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/master/data/atis.json",
        "description": "947 queries, 25 tables",
    },
    "imdb": {
        "url": "https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/master/data/imdb.json",
        "description": "89 queries, 16 tables",
    },

    "yelp": {
        "url": "https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/master/data/yelp.json",
        "description": "110 queries, 7 tables",
    },
}
'''
Unused datasets:
    "restaurants": {
        "url": "https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/master/data/restaurants.json",
        "description": "23 queries, 8 tables",
    },
        "academic": {
        "url": "https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/master/data/academic.json",
        "description": "185 queries, 15 tables",
    },
'''


def make_session() -> requests.Session:
    """Requests session with retries for flaky networks."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "adis-llmsql3-dataset-downloader/1.0"})
    return session


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def analyze_dataset(data: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """Return analysis dict for manifest."""
    if not data:
        return {"name": name, "total": 0, "sql_key": None, "complexity": {}}

    sample = data[0]

    dist = {"simple": 0, "medium": 0, "complex": 0}
    for item in data:
            sql = item.get("sql", "")

    return {
        "name": name,
        "total": len(data),
        "keys": list(sample.keys()),
    }


def download_dataset(
    session: requests.Session,
    name: str,
    url: str,
    out_dir: Path,
    force: bool = False,
) -> Tuple[Optional[List[Dict[str, Any]]], Path, str]:
    """Download dataset from URL unless cached."""
    
    out_path = out_dir / f"{name}.json"
    if out_path.exists() and not force:
        return load_json(out_path), out_path, "cached"

    resp = session.get(url, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    save_json(out_path, data)
    return data, out_path, "downloaded"


def main() -> int:
    print("🚀 Text2SQL Dataset Downloader")
    print("=" * 60)

    output_dir = Path("datasets_source/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()

    manifest = {
        "generated_at_epoch": int(time.time()),
        "datasets": [],
    }

    ok = 0
    for name, info in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"Description: {info['description']}")
        print("=" * 60)

        try:
            data, path, mode = download_dataset(
                session, name, info["url"], output_dir, force=False
            )
            analysis = analyze_dataset(data or [], name)

            print(f"📁 {mode.upper()}: {path}")
            print(f"📊 Total examples: {analysis['total']}")
            
            manifest["datasets"].append(
                {
                    "name": name,
                    "url": info["url"],
                    "description": info["description"],
                    "file": str(path.as_posix()),
                    "mode": mode,
                    **analysis,
                }
            )

            ok += 1
        except Exception as e:
            print(f"❌ Failed for {name}: {e}")

    save_json(Path("datasets_source/manifest.json"), manifest)

    print(f"\n✅ Finished: {ok}/{len(DATASETS)} datasets available")
    print(f"🧾 Manifest: {Path('datasets_source/manifest.json').absolute()}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
