import os

folders = [
    "environment", "agent", "optimization", "visualization",
    "models", "results/layouts", "results/plots", "results/metrics",
    "presentation",
]
files = [
    "README.md", "requirements.txt", ".gitignore",
    "config.py", "main.py", "train.py", "evaluate.py",
    "environment/__init__.py", "environment/factory_env.py",
    "agent/__init__.py", "agent/dqn.py", "agent/replay_buffer.py", "agent/policy.py",
    "optimization/__init__.py", "optimization/flow_matrix.py",
    "optimization/cost_calculator.py", "optimization/baselines.py",
    "visualization/__init__.py", "visualization/factory_plot.py",
    "visualization/training_plot.py",
]

for f in folders:
    os.makedirs(f, exist_ok=True)
for f in files:
    open(f, "a").close()
print("Structure created.")