from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data_utils import (
    classification_metrics,
    ensure_dir,
    load_dataset,
    plot_training_history,
    positive_class_probs_from_logits,
    save_json,
    set_seed,
    split_dataset,
)

MODEL_OUTPUT_DIRS = {
    "bert-base-cased": "artifacts/bert_base",
    "bert-large-cased": "artifacts/bert_large",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for AI-text detection.")
    parser.add_argument("--data_path", default="train_v2_drcat_02.csv")
    parser.add_argument("--model_name", default="bert-base-cased")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--stratify_strategy",
        choices=("label", "label_prompt"),
        default="label_prompt",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument(
        "--final_eval_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    return parser.parse_args()


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    probs = positive_class_probs_from_logits(logits)
    metrics = classification_metrics(labels, probs)
    return {
        "roc_auc": metrics["roc_auc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def maybe_limit_frame(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    try:
        import accelerate  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The `accelerate` package is required for Trainer. Install it with `pip install accelerate`."
        ) from exc

    set_seed(args.seed)

    output_dir = ensure_dir(
        args.output_dir or MODEL_OUTPUT_DIRS.get(args.model_name, "artifacts/bert_model")
    )
    model_dir = ensure_dir(output_dir / "model")

    if torch.cuda.is_available() and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    df = load_dataset(args.data_path)
    train_df, val_df = split_dataset(
        df,
        test_size=args.test_size,
        seed=args.seed,
        stratify_strategy=args.stratify_strategy,
    )
    train_df = maybe_limit_frame(train_df, args.max_train_samples, args.seed)
    val_df = maybe_limit_frame(val_df, args.max_eval_samples, args.seed)

    train_frame = train_df[["row_id", "text", "label", "prompt_name", "source"]].rename(
        columns={"label": "labels"}
    )
    val_frame = val_df[["row_id", "text", "label", "prompt_name", "source"]].rename(
        columns={"label": "labels"}
    )

    hf_train = Dataset.from_pandas(train_frame, preserve_index=False)
    hf_val = Dataset.from_pandas(val_frame, preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )

    def tokenize_function(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_train = hf_train.map(tokenize_function, batched=True)
    tokenized_val = hf_val.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    if train_batch_size is None:
        train_batch_size = 8 if args.model_name == "bert-large-cased" else 16
    if eval_batch_size is None:
        eval_batch_size = train_batch_size

    steps_per_epoch = math.ceil(len(train_df) / train_batch_size)
    total_training_steps = max(
        1,
        math.ceil(steps_per_epoch * args.epochs / args.gradient_accumulation_steps),
    )
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    eval_strategy = "no" if args.final_eval_only else "epoch"
    save_strategy = "no" if args.final_eval_only else "epoch"
    load_best_model = not args.final_eval_only

    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        do_train=True,
        do_eval=not args.final_eval_only,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best_model,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=True,
        skip_memory_metrics=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"Starting training for {args.model_name}...")
    train_result = trainer.train()
    eval_results = trainer.evaluate() if not args.final_eval_only else {}

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    prediction_output = trainer.predict(tokenized_val)
    val_probs = positive_class_probs_from_logits(prediction_output.predictions)
    final_metrics = classification_metrics(val_df["label"], val_probs)
    final_metrics.update(
        {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "final_eval_only": args.final_eval_only,
            "warmup_steps": warmup_steps,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "peak_gpu_memory_mb": round(
                torch.cuda.max_memory_allocated() / (1024**2), 2
            )
            if torch.cuda.is_available()
            else 0.0,
        }
    )

    train_metrics = {
        key: float(value) if isinstance(value, (int, float)) else value
        for key, value in train_result.metrics.items()
    }
    eval_metrics = {
        key: float(value) if isinstance(value, (int, float)) else value
        for key, value in eval_results.items()
    }

    predictions_df = val_df[["row_id", "label", "prompt_name", "source", "text"]].copy()
    predictions_df["ai_probability"] = val_probs
    predictions_df["pred_label"] = (predictions_df["ai_probability"] >= 0.5).astype(int)
    predictions_df["text_snippet"] = predictions_df["text"].str.slice(0, 240)
    predictions_df.drop(columns=["text"]).to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
    )

    history_df = pd.DataFrame(trainer.state.log_history)
    history_df.to_csv(output_dir / "training_log_history.csv", index=False)
    plot_training_history(trainer.state.log_history, output_dir / "training_history.png")

    save_json(vars(args), output_dir / "run_config.json")
    save_json(train_metrics, output_dir / "train_metrics.json")
    save_json(eval_metrics, output_dir / "eval_metrics.json")
    save_json(final_metrics, output_dir / "validation_metrics.json")

    print(f"Validation ROC-AUC for {args.model_name}: {final_metrics['roc_auc']:.4f}")
    print(f"Saved model artifacts to: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
