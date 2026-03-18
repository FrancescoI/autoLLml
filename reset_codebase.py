import os
import shutil
import subprocess


def reset_codebase():
    """Reset codebase to baseline, keep MLFlow runs."""
    
    items = [
        "evaluation_report.json",
        "evaluation_report.md",
        "evaluation_plots",
        "__pycache__",
        "agents/__pycache__",
    ]
    
    for item in items:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
            print(f"[*] Deleted: {item}")
    
    # Restore baseline dynamic_features.py from git
    result = subprocess.run(
        ["git", "checkout", "dynamic_features.py"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("[*] Restored: dynamic_features.py from git")
    else:
        print(f"[!] Could not restore dynamic_features.py: {result.stderr}")
    
    print("\n[+] Codebase reset complete")
    print("[*] MLFlow runs preserved in: mlruns/")
    print("[*] To start new experiment: python main.py --iterations 5")


if __name__ == "__main__":
    reset_codebase()
