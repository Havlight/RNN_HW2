from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from data_utils import (
    ensure_dir,
    load_dataset,
    positive_class_probs_from_logits,
    save_json,
    select_human_attack_samples,
    set_seed,
    split_dataset,
)

ATTACK_PROMPTS = [
    "Rewrite the following essay so it sounds more natural and human-written.",
    "Rewrite this essay as if a high school student wrote it in their own words.",
    "Paraphrase the following text while keeping the ideas simple and natural.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local-LLM adversarial rewrite attacks.")
    parser.add_argument("--data_path", default="train_v2_drcat_02.csv")
    parser.add_argument("--detector_dir", default="artifacts/bert_base/model")
    parser.add_argument(
        "--gen_model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument("--output_dir", default="artifacts/attacks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--stratify_strategy",
        choices=("label", "label_prompt"),
        default="label_prompt",
    )
    parser.add_argument("--num_essays", type=int, default=8)
    parser.add_argument("--variants_per_essay", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--detector_batch_size", type=int, default=8)
    return parser.parse_args()


def build_generation_prompt(tokenizer: AutoTokenizer, instruction: str, essay: str) -> str:
    messages = [
        {"role": "system", "content": "You are a careful writing assistant."},
        {
            "role": "user",
            "content": f"{instruction}\n\nEssay:\n{essay}",
        },
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError:
            pass
    return (
        "System: You are a careful writing assistant.\n"
        f"User: {instruction}\n\nEssay:\n{essay}\nAssistant:"
    )


def extract_generated_text(generation_output: list[dict[str, object]]) -> str:
    if not generation_output:
        return ""
    first_item = generation_output[0]
    if isinstance(first_item, dict):
        generated = first_item.get("generated_text")
        if isinstance(generated, str):
            return generated.strip()
    return str(first_item).strip()


def score_texts(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int,
) -> list[float]:
    device = next(model.parameters()).device
    probabilities: list[float] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu().numpy()
        probabilities.extend(positive_class_probs_from_logits(logits).tolist())

    return probabilities


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    df = load_dataset(args.data_path)
    _, val_df = split_dataset(
        df,
        test_size=args.test_size,
        seed=args.seed,
        stratify_strategy=args.stratify_strategy,
    )
    attack_samples = select_human_attack_samples(val_df, num_samples=args.num_essays)
    attack_samples.to_csv(output_dir / "selected_human_essays.csv", index=False)

    generator_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    generator_kwargs: dict[str, object] = {
        "model": args.gen_model,
        "dtype": generator_dtype,
    }
    if torch.cuda.is_available():
        generator_kwargs["device_map"] = "auto"

    print(f"Loading generator model: {args.gen_model}")
    generator = pipeline("text-generation", **generator_kwargs)
    generator_tokenizer = generator.tokenizer

    rewrite_rows: list[dict[str, object]] = []
    prompt_pool = ATTACK_PROMPTS[: max(1, min(args.variants_per_essay, len(ATTACK_PROMPTS)))]

    for essay in attack_samples.itertuples(index=False):
        for prompt_index, prompt_text in enumerate(prompt_pool, start=1):
            rendered_prompt = build_generation_prompt(generator_tokenizer, prompt_text, essay.text)
            output = generator(
                rendered_prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                return_full_text=False,
            )
            rewritten_text = extract_generated_text(output)
            rewrite_rows.append(
                {
                    "row_id": int(essay.row_id),
                    "label": int(essay.label),
                    "prompt_name": essay.prompt_name,
                    "source": essay.source,
                    "word_count": int(essay.word_count),
                    "attack_prompt_id": prompt_index,
                    "attack_prompt": prompt_text,
                    "original_text": essay.text,
                    "rewritten_text": rewritten_text,
                }
            )

    rewrites_df = pd.DataFrame(rewrite_rows)
    rewrites_df["rewritten_text_snippet"] = rewrites_df["rewritten_text"].str.slice(0, 240)
    rewrites_df.to_csv(output_dir / "generated_rewrites_raw.csv", index=False)

    del generator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    detector_dir = Path(args.detector_dir)
    if not detector_dir.exists():
        raise FileNotFoundError(
            f"Detector directory does not exist: {detector_dir}. "
            "Run BERT.py first or pass --detector_dir."
        )

    print(f"Loading detector from: {detector_dir}")
    detector = AutoModelForSequenceClassification.from_pretrained(detector_dir)
    detector_tokenizer = AutoTokenizer.from_pretrained(detector_dir)
    detector_device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = detector.to(detector_device).eval()

    original_probs = score_texts(
        detector,
        detector_tokenizer,
        rewrites_df["original_text"].tolist(),
        batch_size=args.detector_batch_size,
    )
    rewritten_probs = score_texts(
        detector,
        detector_tokenizer,
        rewrites_df["rewritten_text"].tolist(),
        batch_size=args.detector_batch_size,
    )

    rewrites_df["original_ai_probability"] = original_probs
    rewrites_df["rewritten_ai_probability"] = rewritten_probs
    rewrites_df["rewritten_pred_label"] = (
        rewrites_df["rewritten_ai_probability"] >= 0.5
    ).astype(int)
    rewrites_df["fooled_or_not"] = rewrites_df["rewritten_pred_label"].eq(0)
    rewrites_df["delta_ai_probability"] = (
        rewrites_df["rewritten_ai_probability"] - rewrites_df["original_ai_probability"]
    )
    rewrites_df.to_csv(output_dir / "attack_results.csv", index=False)

    summary = {
        "generator_model": args.gen_model,
        "detector_dir": str(detector_dir),
        "num_essays": int(args.num_essays),
        "variants_per_essay": int(len(prompt_pool)),
        "total_attack_samples": int(len(rewrites_df)),
        "mean_original_ai_probability": round(
            float(rewrites_df["original_ai_probability"].mean()), 6
        ),
        "mean_rewritten_ai_probability": round(
            float(rewrites_df["rewritten_ai_probability"].mean()), 6
        ),
        "successful_fool_count": int(rewrites_df["fooled_or_not"].sum()),
        "successful_fool_rate": round(float(rewrites_df["fooled_or_not"].mean()), 6),
    }
    save_json(summary, output_dir / "attack_summary.json")

    print(f"Attack success rate: {summary['successful_fool_rate']:.2%}")
    print(f"Saved attack artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
