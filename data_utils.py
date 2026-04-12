from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = ("text", "label", "prompt_name", "source")
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(
        json.dumps(_to_builtin(data), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Expected at least {list(REQUIRED_COLUMNS)}."
        )

    df = df.copy()
    df["row_id"] = np.arange(len(df))
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["prompt_name"] = df["prompt_name"].fillna("unknown").astype(str)
    df["source"] = df["source"].fillna("unknown").astype(str)
    return df


def build_split_key(df: pd.DataFrame, strategy: str = "label_prompt") -> pd.Series:
    if strategy == "label":
        return df["label"].astype(str)
    if strategy == "label_prompt":
        return df["label"].astype(str) + "__" + df["prompt_name"].astype(str)
    raise ValueError(f"Unsupported stratify strategy: {strategy}")


def split_dataset(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    stratify_strategy: str = "label_prompt",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_key = build_split_key(df, stratify_strategy)
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=split_key,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    char_count: list[int] = []
    word_count: list[int] = []
    unique_word_count: list[int] = []
    type_token_ratio: list[float] = []

    for text in frame["text"]:
        tokens = str(text).split()
        lowered_tokens = [token.lower() for token in tokens]
        token_count = len(tokens)
        unique_count = len(set(lowered_tokens))

        char_count.append(len(text))
        word_count.append(token_count)
        unique_word_count.append(unique_count)
        type_token_ratio.append(unique_count / token_count if token_count else 0.0)

    frame["char_count"] = char_count
    frame["word_count"] = word_count
    frame["unique_word_count"] = unique_word_count
    frame["type_token_ratio"] = type_token_ratio
    return frame


def _series_summary(series: pd.Series) -> dict[str, float | int]:
    return {
        "count": int(series.count()),
        "mean": round(float(series.mean()), 4),
        "median": round(float(series.median()), 4),
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
    }


def summarize_dataset(feature_df: pd.DataFrame) -> dict[str, Any]:
    label_counts = feature_df["label"].value_counts().sort_index()
    by_label_rows: list[dict[str, Any]] = []

    for label, group in feature_df.groupby("label"):
        by_label_rows.append(
            {
                "label": int(label),
                "samples": int(len(group)),
                "char_count_mean": round(float(group["char_count"].mean()), 4),
                "word_count_mean": round(float(group["word_count"].mean()), 4),
                "unique_word_count_mean": round(
                    float(group["unique_word_count"].mean()), 4
                ),
                "type_token_ratio_mean": round(
                    float(group["type_token_ratio"].mean()), 4
                ),
            }
        )

    return {
        "num_rows": int(len(feature_df)),
        "columns": list(feature_df.columns),
        "label_counts": {int(key): int(value) for key, value in label_counts.items()},
        "prompt_count": int(feature_df["prompt_name"].nunique()),
        "source_count": int(feature_df["source"].nunique()),
        "char_count": _series_summary(feature_df["char_count"]),
        "word_count": _series_summary(feature_df["word_count"]),
        "unique_word_count": _series_summary(feature_df["unique_word_count"]),
        "type_token_ratio": _series_summary(feature_df["type_token_ratio"]),
        "whitespace_tokens_gt_512": int((feature_df["word_count"] > 512).sum()),
        "whitespace_tokens_gt_512_ratio": round(
            float((feature_df["word_count"] > 512).mean()), 4
        ),
        "by_label": by_label_rows,
    }


def label_feature_table(feature_df: pd.DataFrame) -> pd.DataFrame:
    table = (
        feature_df.groupby("label")[
            ["char_count", "word_count", "unique_word_count", "type_token_ratio"]
        ]
        .agg(["mean", "median", "min", "max"])
        .round(4)
    )
    table.columns = ["_".join(column).strip() for column in table.columns.to_flat_index()]
    return table.reset_index()


def classification_metrics(
    labels: np.ndarray | list[int] | pd.Series,
    positive_probs: np.ndarray | list[float] | pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    label_array = np.asarray(labels)
    prob_array = np.asarray(positive_probs, dtype=float)
    pred_array = (prob_array >= threshold).astype(int)
    conf = confusion_matrix(label_array, pred_array, labels=[0, 1])

    return {
        "roc_auc": round(float(roc_auc_score(label_array, prob_array)), 6),
        "accuracy": round(float(accuracy_score(label_array, pred_array)), 6),
        "precision": round(
            float(precision_score(label_array, pred_array, zero_division=0)), 6
        ),
        "recall": round(
            float(recall_score(label_array, pred_array, zero_division=0)), 6
        ),
        "f1": round(float(f1_score(label_array, pred_array, zero_division=0)), 6),
        "confusion_matrix": conf.tolist(),
        "threshold": float(threshold),
    }


def positive_class_probs_from_logits(logits: np.ndarray | list[list[float]]) -> np.ndarray:
    logits_array = np.asarray(logits, dtype=float)
    if logits_array.ndim != 2 or logits_array.shape[1] < 2:
        raise ValueError("Expected logits with shape [batch_size, 2] for binary classification.")
    shifted = logits_array - logits_array.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return probs[:, 1]


def plot_label_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    counts = df["label"].value_counts().sort_index()
    labels = ["Human (0)", "AI (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["#4C72B0", "#DD8452"])
    plt.ylabel("Samples")
    plt.title("Label Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_distribution(
    feature_df: pd.DataFrame,
    feature_name: str,
    output_path: str | Path,
    title: str,
    xlabel: str,
    bins: int = 50,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for label, color, legend in (
        (0, "#4C72B0", "Human"),
        (1, "#DD8452", "AI"),
    ):
        subset = feature_df.loc[feature_df["label"] == label, feature_name]
        plt.hist(
            subset,
            bins=bins,
            alpha=0.55,
            density=True,
            label=legend,
            color=color,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_curve(
    labels: np.ndarray | list[int] | pd.Series,
    positive_probs: np.ndarray | list[float] | pd.Series,
    output_path: str | Path,
    title: str = "ROC Curve",
) -> None:
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels, positive_probs)
    auc_value = roc_auc_score(labels, positive_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_value:.4f}", color="#4C72B0")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_history(log_history: list[dict[str, Any]], output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    train_steps: list[float] = []
    train_loss: list[float] = []
    eval_steps: list[float] = []
    eval_loss: list[float] = []
    eval_auc: list[float] = []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])
            if "eval_roc_auc" in entry:
                eval_auc.append(entry["eval_roc_auc"])

    if not train_steps and not eval_steps:
        return

    plt.figure(figsize=(8, 5))
    if train_steps:
        plt.plot(train_steps, train_loss, label="train_loss", color="#4C72B0")
    if eval_steps and eval_loss:
        plt.plot(eval_steps, eval_loss, label="eval_loss", color="#DD8452")
    if eval_steps and eval_auc:
        plt.plot(eval_steps, eval_auc, label="eval_roc_auc", color="#55A868")

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def select_human_attack_samples(
    val_df: pd.DataFrame,
    num_samples: int,
    min_words: int | None = None,
    max_words: int | None = None,
) -> pd.DataFrame:
    human_df = add_text_features(val_df[val_df["label"] == 0].copy())
    if human_df.empty:
        raise ValueError("Validation split does not contain any human-written essays.")

    if min_words is not None:
        human_df = human_df.loc[human_df["word_count"] >= min_words].copy()
    if max_words is not None:
        human_df = human_df.loc[human_df["word_count"] <= max_words].copy()
    if human_df.empty:
        raise ValueError(
            "No human-written essays remain after applying the word-count filter."
        )
    if len(human_df) < num_samples:
        raise ValueError(
            f"Only {len(human_df)} eligible human essays are available, "
            f"but {num_samples} were requested."
        )

    human_df = human_df.sort_values(["word_count", "row_id"]).reset_index(drop=True)
    positions = np.linspace(0, len(human_df) - 1, num=num_samples, dtype=int)
    unique_positions = sorted(set(int(position) for position in positions))
    selected = human_df.iloc[unique_positions].copy().reset_index(drop=True)

    if len(selected) < num_samples:
        needed = num_samples - len(selected)
        remaining = human_df.drop(index=unique_positions)
        filler = remaining.sample(n=needed, random_state=DEFAULT_SEED)
        selected = pd.concat([selected, filler], ignore_index=True)

    return selected.reset_index(drop=True)
