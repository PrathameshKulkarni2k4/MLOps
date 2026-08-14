import argparse
import sys

def handle_train(args):
    print(f"[TRAIN] Data: {args.data_path} | Epochs: {args.epochs} | LR: {args.lr}")

def handle_eval(args):
    print(f"[EVAL] Model: {args.model_path} | Batch Size: {args.batch_size}")

def main():
    parser = argparse.ArgumentParser(
        description="MLOps Training & Evaluation Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Run model training")
    train_parser.add_argument("--data-path", type=str, required=True, help="Path to training data")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.set_defaults(func=handle_train)

    # Eval subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--model-path", type=str, required=True, help="Path to saved model binary")
    eval_parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    eval_parser.set_defaults(func=handle_eval)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()