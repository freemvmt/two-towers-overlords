"""
Main entry point for training the two-towers document retrieval model.
"""

import argparse
import wandb
from training import run_training


def main():
    """Main function to run training with command-line arguments."""
    parser = argparse.ArgumentParser(description="Train two-towers document retrieval model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--max-samples", type=int, default=10_000, help="Maximum number of training samples (for faster development)"
    )
    parser.add_argument("--projection-dim", type=int, default=128, help="Project dimension for each tower")
    parser.add_argument("--margin", type=float, default=0.1, help="Margin for triplet loss")
    parser.add_argument("--project-name", type=str, default="two-towers-retrieval", help="Wandb project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep with wandb")

    args = parser.parse_args()

    if args.sweep:
        run_sweep(args.project_name)
        return

    print("Two-Towers Document Retrieval Model")
    print("=" * 40)
    print("Training configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Projection dimension: {args.projection_dim}")
    print(f"  Margin: {args.margin}")
    print(f"  Wandb enabled: {not args.no_wandb}")
    print()

    # Run training
    trained_model = run_training(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        projection_dim=args.projection_dim,
        margin=args.margin,
        project_name=args.project_name,
        use_wandb=not args.no_wandb,
    )

    print("\nTraining completed!")
    print(f"Model trained successfully with {sum(p.numel() for p in trained_model.parameters())} parameters.")


def sweep_train():
    """Training function for wandb sweep."""
    # Initialize wandb
    wandb.init()

    # Get hyperparameters from wandb config
    config = wandb.config

    # Run training with sweep parameters
    trained_model = run_training(
        num_epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_samples=config.max_samples,
        projection_dim=config.projection_dim,
        margin=config.margin,
        project_name="two-towers-retrieval-sweep",
        use_wandb=True,
    )

    return trained_model


def run_sweep(project_name: str = "two-towers-retrieval-sweep"):
    """Run hyperparameter sweep with wandb."""
    sweep_config = {
        "method": "bayes",  # Can be 'grid', 'random', or 'bayes'
        "metric": {"name": "val_ndcg_10", "goal": "maximize"},
        "parameters": {
            "epochs": {"values": [3, 5, 8]},
            "batch_size": {"values": [16, 32, 64, 128]},
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
            "max_samples": {"values": [1000, 2000, 5000]},
            "projection_dim": {"values": [64, 128, 256]},
        },
    }

    print("Starting hyperparameter sweep...")
    print(f"Sweep configuration: {sweep_config}")

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Sweep ID: {sweep_id}")
    print("Run the following command to start agents:")
    print(f"wandb agent {sweep_id}")

    # Optionally run a single agent automatically
    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        project=project_name,
        count=5,
    )


if __name__ == "__main__":
    main()
