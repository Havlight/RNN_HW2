from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data_utils import (
    add_text_features,
    classification_metrics,
    ensure_dir,
    label_feature_table,
    load_dataset,
    plot_feature_distribution,
    plot_label_distribution,
    plot_roc_curve,
    save_json,
    select_human_attack_samples,
    set_seed,
    split_dataset,
    summarize_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--data_path", default="train_v2_drcat_02.csv")
    parser.add_argument("--output_dir", default="artifacts/baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--stratify_strategy",
        choices=("label", "label_prompt"),
        default="label_prompt",
    )
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--class_weight", default=None)
    parser.add_argument(
        "--sublinear_tf",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    df = load_dataset(args.data_path)
    feature_df = add_text_features(df)
    train_df, val_df = split_dataset(
        df,
        test_size=args.test_size,
        seed=args.seed,
        stratify_strategy=args.stratify_strategy,
    )

    dataset_summary = summarize_dataset(feature_df)
    dataset_summary["split"] = {
        "seed": args.seed,
        "test_size": args.test_size,
        "stratify_strategy": args.stratify_strategy,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
    }
    save_json(dataset_summary, output_dir / "eda_summary.json")
    label_feature_table(feature_df).to_csv(output_dir / "eda_by_label.csv", index=False)

    plot_label_distribution(feature_df, output_dir / "label_distribution.png")
    plot_feature_distribution(
        feature_df,
        feature_name="word_count",
        output_path=output_dir / "word_count_distribution.png",
        title="Word Count Distribution",
        xlabel="Whitespace Token Count",
    )
    plot_feature_distribution(
        feature_df,
        feature_name="unique_word_count",
        output_path=output_dir / "unique_word_count_distribution.png",
        title="Vocabulary Richness Distribution",
        xlabel="Unique Whitespace Tokens",
    )
    plot_feature_distribution(
        feature_df,
        feature_name="type_token_ratio",
        output_path=output_dir / "type_token_ratio_distribution.png",
        title="Type-Token Ratio Distribution",
        xlabel="Type-Token Ratio",
    )

    print("Training TF-IDF baseline...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        sublinear_tf=args.sublinear_tf,
    )
    x_train = vectorizer.fit_transform(train_df["text"])
    x_val = vectorizer.transform(val_df["text"])

    classifier = LogisticRegression(
        solver="liblinear",
        max_iter=args.max_iter,
        class_weight=args.class_weight,
    )
    classifier.fit(x_train, train_df["label"])

    val_probs = classifier.predict_proba(x_val)[:, 1]
    metrics = classification_metrics(val_df["label"], val_probs)
    metrics["model"] = "tfidf_logistic_regression"

    predictions_df = val_df[["row_id", "label", "prompt_name", "source", "text"]].copy()
    predictions_df["ai_probability"] = val_probs
    predictions_df["pred_label"] = (predictions_df["ai_probability"] >= 0.5).astype(int)
    predictions_df["text_snippet"] = predictions_df["text"].str.slice(0, 240)
    predictions_df.drop(columns=["text"]).to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
    )

    plot_roc_curve(
        val_df["label"],
        val_probs,
        output_dir / "baseline_roc_curve.png",
        title="TF-IDF Baseline ROC Curve",
    )

    model_bundle = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "metrics": metrics,
    }
    joblib.dump(model_bundle, output_dir / "baseline_model.joblib")
    save_json(metrics, output_dir / "baseline_metrics.json")

    attack_candidates = select_human_attack_samples(val_df, num_samples=8)[
        ["row_id", "label", "prompt_name", "source", "word_count", "text"]
    ].copy()
    attack_candidates["text_snippet"] = attack_candidates["text"].str.slice(0, 240)
    attack_candidates.drop(columns=["text"]).to_csv(
        output_dir / "attack_candidates.csv",
        index=False,
    )

    print(f"Baseline TF-IDF ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Saved baseline artifacts to: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
