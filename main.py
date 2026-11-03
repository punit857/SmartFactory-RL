import subprocess, sys

for script in ["test_env.py", "run_baselines.py", "evaluate.py"]:
    print(f"\n>>> Running {script}")
    subprocess.run([sys.executable, script], check=True)