import subprocess
import sys
import time
from pathlib import Path

# Λίστα με τα ονόματα των datasets που θέλεις να τρέξεις.
# Το script υποθέτει ότι τα αρχεία json βρίσκονται στο datasets_source/data/<name>.json
DATASETS = [
    "advising",
    "atis"#, imdb
]

def main():
    # Το μονοπάτι προς το script που ανέβασες
    baseline_script = Path("scripts/run_qwen_baseline.py")
    
    if not baseline_script.exists():
        print(f"❌ Error: Δεν βρέθηκε το αρχείο {baseline_script}")
        return

    print("🚀 Starting Batch Benchmark Run...")
    print(f"Datasets to run: {', '.join(DATASETS)}")
    print("="*60)

    for db_name in DATASETS:
        dataset_path = f"datasets_source/data/{db_name}.json"
        
        # Ελέγχουμε αν υπάρχει το dataset πριν ξεκινήσουμε
        if not Path(dataset_path).exists():
            print(f"⚠️ Skipping {db_name}: Δεν βρέθηκε το αρχείο {dataset_path}")
            continue

        print(f"\n▶️ Running benchmark for: {db_name.upper()}")
        
        # Εντολή: python scripts/run_qwen_baseline.py --dataset ... --limit_entries 50
        cmd = [
            sys.executable, str(baseline_script),
            "--dataset", dataset_path,
            "--limit_entries", "50",   # Τρέχουμε 50 ερωτήσεις για κάθε βάση
            "--rdbms", "mysql"         # Μπορείς να βάλεις "both" αν θες και MariaDB
        ]

        try:
            # Καλούμε το baseline script ως υπο-διεργασία
            subprocess.run(cmd, check=True)
            print(f"✅ Finished {db_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed processing {db_name}. Error code: {e.returncode}")
        except Exception as e:
            print(f"❌ Unexpected error on {db_name}: {e}")
        
        # Μικρή παύση 2 δευτερολέπτων για να ηρεμήσει ο επεξεργαστής
        time.sleep(2)

    print("\n" + "="*60)
    print("🎉 Batch Run Complete! Check the 'results' folder.")

if __name__ == "__main__":
    main()