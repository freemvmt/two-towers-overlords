"""
Main entry point for training the two-towers document retrieval model.
"""

import argparse
import json
from math import log10
import os
from datetime import datetime

# Heavy imports moved to lazy loading for faster startup
# import torch
# from torchinfo import summary
# import wandb
# from training import run_training


MODELS_DIR = "models"
MODEL_FILENAME_BASE_TEMPLATE = "e{epochs}.lr{learning_rate}.d{projection_dim}.m{margin}"


def main():
    """Main function to run training with command-line arguments."""
    parser = argparse.ArgumentParser(description="Train two-towers document retrieval model")
    # size of dataset to use (-1 for full dataset) and batch size
    parser.add_argument(
        "--max-samples", type=int, default=100_000, help="Maximum number of training samples (for faster development)"
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training")

    # hyperparams proper (assume a small but effective run as default, modelled on iconic sweep)
    parser.add_argument("--epochs", type=int, default=9, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--projection-dim", type=int, default=128, help="Project dimension for each tower")
    parser.add_argument("--margin", type=float, default=0.3, help="Margin for triplet loss")

    # logistics / switches
    parser.add_argument("--project-name", type=str, default="two-towers-retrieval", help="Wandb project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--no-save", action="store_true", help="Don't save weights after training")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep with wandb")
    parser.add_argument(
        "--no-comprehensive-test", action="store_true", help="Skip comprehensive testing after training"
    )

    # GPU optimization
    parser.add_argument(
        "--accumulation-steps", type=int, default=2, help="Gradient accumulation steps for larger effective batch size"
    )
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")

    args = parser.parse_args()

    # Early exit for sweep
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
    print(f"  Comprehensive testing: {not args.no_comprehensive_test}")
    print()

    # Import heavy libraries only when we need them for training
    print("Loading PyTorch and training modules...")
    import torch
    from torchinfo import summary
    from training import run_training

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
        accumulation_steps=args.accumulation_steps,
        use_mixed_precision=not args.no_mixed_precision,
        num_workers=args.num_workers,
        run_comprehensive_test=not args.no_comprehensive_test,
    )

    # Print model summary and save state
    print("\n✅ Finished training!")
    print(f"   Model trained successfully with {sum(p.numel() for p in trained_model.parameters())} parameters.")

    # Generate and display model summary
    model_summary = summary(trained_model, verbose=0)  # verbose=0 to get return object
    print(model_summary)

    if not args.no_save:
        try:
            # Ensure the models directory exists
            os.makedirs(MODELS_DIR, exist_ok=True)
            # encode the filename with the core hyperparams for clarity
            base_name = MODEL_FILENAME_BASE_TEMPLATE.format(
                epochs=args.epochs,
                learning_rate=abs(int(log10(args.learning_rate))),
                projection_dim=args.projection_dim,
                margin=int(args.margin * 10),
            )

            # Save model weights
            weights_path = os.path.join(MODELS_DIR, f"{base_name}.pt")
            torch.save(trained_model.state_dict(), weights_path)
            print(f"\n✅ Model weights saved to: {weights_path}")

            # Save model summary as text
            summary_txt_path = os.path.join(MODELS_DIR, f"{base_name}_summary.txt")
            with open(summary_txt_path, "w") as f:
                f.write(str(model_summary))
            print(f"✅ Model summary saved to: {summary_txt_path}")

            # Save detailed model info as JSON
            model_info = {
                "training_config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "max_samples": args.max_samples,
                    "projection_dim": args.projection_dim,
                    "margin": args.margin,
                    "accumulation_steps": args.accumulation_steps,
                    "num_workers": args.num_workers,
                },
                "model_stats": {
                    "total_params": sum(p.numel() for p in trained_model.parameters()),
                    "trainable_params": sum(p.numel() for p in trained_model.parameters() if p.requires_grad),
                    "model_size_mb": sum(p.numel() * p.element_size() for p in trained_model.parameters())
                    / (1024 * 1024),
                },
                "files": {
                    "weights": f"{base_name}.pt",
                    "summary": f"{base_name}_summary.txt",
                    "info": f"{base_name}_info.json",
                },
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
            }

            # Add summary statistics if available
            if hasattr(model_summary, "total_params"):
                model_info["torchinfo_summary"] = {
                    "total_params": model_summary.total_params if hasattr(model_summary, "total_params") else None,
                    "trainable_params": (
                        model_summary.trainable_params if hasattr(model_summary, "trainable_params") else None
                    ),
                    "total_mult_adds": (
                        model_summary.total_mult_adds if hasattr(model_summary, "total_mult_adds") else None
                    ),
                }

            info_json_path = os.path.join(MODELS_DIR, f"{base_name}_info.json")
            with open(info_json_path, "w") as f:
                json.dump(model_info, f, indent=2)
            print(f"✅ Model info saved to: {info_json_path}")

        except Exception as e:
            print(f"\n❌ Error saving model: {e}")
            print("Training completed successfully, but model saving failed.")
    else:
        print("\nModel state not saved (--no-save passed as arg)")


def sweep_train():
    """Training function for wandb sweep."""
    import wandb
    from training import run_training

    # Initialize wandb for the sweep run if not already done
    if not wandb.run:
        wandb.init()

    # Get hyperparameters from wandb config
    config = wandb.config

    # Run training with sweep parameters
    # assume we're running on beefy GPU (e.g. RTX 3090), so default to high accumulation steps / workers
    print("Running training with hyperparameters:")
    trained_model = run_training(
        num_epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_samples=config.max_samples,
        projection_dim=config.projection_dim,
        margin=config.margin,
        project_name="two-towers-retrieval",
        use_wandb=True,
        accumulation_steps=config.get("accumulation_steps", 2),
        num_workers=config.get("num_workers", 4),
        run_comprehensive_test=config.get("run_comprehensive_test", True),  # Enable by default for sweeps
    )

    return trained_model


def run_sweep(project_name: str = "two-towers-retrieval"):
    """Run hyperparameter sweep with wandb."""
    sweep_config = {
        "method": "grid",  # Can be 'grid', 'random', or 'bayes'
        "metric": {"name": "final_ndcg_10", "goal": "maximize"},
        "parameters": {
            "margin": {"values": [0.3]},
            "epochs": {"values": [15, 20, 25]},
            "batch_size": {"values": [1024]},
            "learning_rate": {"values": [1e-4]},
            "max_samples": {"values": [-1]},  # -1 means use full dataset
            "projection_dim": {"values": [384, 512]},
        },
    }

    print("Starting hyperparameter sweep...")
    print(f"Sweep configuration: {sweep_config}")

    # Import wandb here to avoid unnecessary dependency if not running a sweep
    import wandb

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Sweep ID: {sweep_id}")
    print("Run the following command to start agents:")
    print(f"wandb agent {sweep_id}")

    # Optionally run a single agent automatically
    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        project=project_name,
        count=6,
    )


if __name__ == "__main__":
    main()
