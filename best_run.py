import os
import shutil
import json

BEST_DIR = "best_run"


def save_best():
    """Save current dynamic_features.py + metrics to best_run/ (local only)."""
    os.makedirs(BEST_DIR, exist_ok=True)
    
    if os.path.exists("dynamic_features.py"):
        shutil.copy2("dynamic_features.py", f"{BEST_DIR}/dynamic_features.py")
        print(f"[*] Saved: dynamic_features.py -> {BEST_DIR}/")
    
    if os.path.exists("evaluation_report.json"):
        shutil.copy2("evaluation_report.json", f"{BEST_DIR}/evaluation_report.json")
        print(f"[*] Saved: evaluation_report.json -> {BEST_DIR}/")
        
        with open("evaluation_report.json", "r") as f:
            report = json.load(f)
        score = report.get("score_mean")
        if score is not None:
            print(f"[*] Best model score: {score:.4f}")
    
    print(f"\n[+] Best run saved to: {BEST_DIR}/")
    print("[*] This directory is local-only (not committed to git)")


def restore_best():
    """Copy best_run/dynamic_features.py to working directory."""
    src = f"{BEST_DIR}/dynamic_features.py"
    if os.path.exists(src):
        shutil.copy2(src, "dynamic_features.py")
        print(f"[*] Restored: {BEST_DIR}/dynamic_features.py -> ./dynamic_features.py")
    else:
        print(f"[!] No saved model found in {BEST_DIR}/")
        print("[*] Run experiments first, then save best run")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        save_best()
    elif len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_best()
    else:
        print("Usage: python best_run.py [save|restore]")