"""
Wandb sweep configurations for hyperparameter tuning.
"""

import wandb

from main import sweep_train

# Basic sweep configuration
BASIC_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_ndcg_10", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [3, 5, 8]},
        "batch_size": {"values": [16, 32, 64, 128]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "max_samples": {"values": [1000, 2000, 5000]},
        "projection_dim": {"values": [128]},
        "margin": {"values": [0.05, 0.1, 0.2]},
    },
}

# Aggressive sweep for extensive exploration
EXTENSIVE_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_ndcg_10", "goal": "maximize"},
    "parameters": {
        "epochs": {"distribution": "int_uniform", "min": 2, "max": 15},
        "batch_size": {"values": [2, 4, 8, 16, 32, 64, 128, 256]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        "max_samples": {"values": [500, 1000, 2000, 5000, 10000]},
        "projection_dim": {"values": [128, 256]},
        "margin": {"distribution": "uniform", "min": 0.01, "max": 0.5},
    },
}

# Quick sweep for development
QUICK_SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "val_ndcg_10", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [2, 3]},
        "batch_size": {"values": [16, 32]},
        "learning_rate": {"values": [1e-4, 5e-4]},
        "max_samples": {"value": 1000},
        "projection_dim": {"value": 64},
        "margin": {"values": [0.1, 0.2]},
    },
}


def run_custom_sweep(config_name: str = "basic", project_name: str = "two-towers-retrieval", count: int = 10):
    """Run a custom sweep with predefined configurations."""

    configs = {"basic": BASIC_SWEEP_CONFIG, "extensive": EXTENSIVE_SWEEP_CONFIG, "quick": QUICK_SWEEP_CONFIG}

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

    sweep_config = configs[config_name]

    print(f"Running {config_name} sweep with {count} runs...")
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    print(f"Sweep ID: {sweep_id}")
    print(f"Dashboard: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")

    # Run sweep agents
    wandb.agent(sweep_id, sweep_train, count=count)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run wandb sweeps")
    parser.add_argument(
        "--config", choices=["basic", "extensive", "quick"], default="basic", help="Sweep configuration to use"
    )
    parser.add_argument("--project", default="two-towers-retrieval-sweep", help="Wandb project name")
    parser.add_argument("--count", type=int, default=10, help="Number of sweep runs")

    args = parser.parse_args()

    run_custom_sweep(args.config, args.project, args.count)
