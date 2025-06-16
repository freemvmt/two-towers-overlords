"""
Example script demonstrating wandb integration with the two-towers model.
"""

from training import run_training


def demo_wandb_integration():
    """Demo script showing wandb features."""

    print("🔥 Two-Towers Model with Wandb Integration")
    print("=" * 50)

    # Note: This would require wandb login in a real scenario
    print("💡 To use wandb, first run: wandb login")
    print()

    print("📊 Available training options:")
    print("1. Basic training with wandb logging:")
    print("   uv run python main.py --epochs 2 --max-samples 500")
    print()

    print("2. Training without wandb:")
    print("   uv run python main.py --no-wandb --epochs 2 --max-samples 500")
    print()

    print("3. Quick hyperparameter sweep:")
    print("   uv run python sweep_config.py --config quick --count 3")
    print()

    print("4. Run built-in sweep:")
    print("   uv run python main.py --sweep")
    print()

    print("🎯 Key wandb features integrated:")
    print("• Automatic experiment tracking")
    print("• Real-time loss and metrics logging")
    print("• Model gradient and weight visualization")
    print("• Hyperparameter sweep optimization")
    print("• Configuration and artifact management")
    print()

    print("📈 Metrics tracked:")
    print("• batch_loss: Training loss per batch")
    print("• avg_train_loss: Average training loss per epoch")
    print("• val_ndcg_10: Validation NDCG@10 score")
    print("• total_parameters: Model size")
    print()

    # Demonstrate offline training (without wandb login)
    print("🚀 Running demo training (offline mode)...")
    print("   This will train a small model without wandb logging")

    try:
        model = run_training(
            num_epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            max_samples=100,
            use_wandb=False,  # Disable wandb for demo
        )
        print("✅ Demo training completed successfully!")
        print(f"   Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("   This is expected if datasets are not available")


if __name__ == "__main__":
    demo_wandb_integration()
