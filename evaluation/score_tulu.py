"""
Score Tulu examples from each source by difficulty (loss under the base model).
Produces one JSONL file per source for use in weighted sampling during training.

Usage:
    python evaluation/score_tulu.py
"""

import json
import numpy as np
import tinker
from datasets import load_dataset
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm import tqdm

MODEL = "meta-llama/Llama-3.1-8B"
BATCH_SIZE = 8

# Maps source name -> output file (matches source constants in filter_tulu_examples)
SOURCES = {
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980": "tulu_source1_scored.jsonl",
    "ai2-adapt-dev/no_robots_converted":                    "tulu_source2_scored.jsonl",
    "allenai/tulu-3-sft-personas-math-grade":               "tulu_source3_scored.jsonl",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k":       "tulu_source4_scored.jsonl",
}


def example_to_convo(example):
    """Extract last user/assistant pair from a Tulu messages example."""
    messages = example.get("messages", [])
    pairs = [
        (messages[i]["content"], messages[i + 1]["content"])
        for i in range(len(messages) - 1)
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant"
    ]
    if not pairs:
        return None
    question, answer = pairs[-1]
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def score_examples(examples, tc, renderer):
    """Run batched forward passes and return examples with difficulty_score attached."""
    scored = []
    for i in tqdm(range(0, len(examples), BATCH_SIZE), desc="  Scoring"):
        batch = examples[i : i + BATCH_SIZE]
        datums = []
        valid_indices = []

        for idx, ex in enumerate(batch):
            convo = example_to_convo(ex)
            if convo is None:
                continue
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=512,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            datums.append(datum)
            valid_indices.append(idx)

        if not datums:
            continue

        fwd_result = tc.forward_backward(datums, loss_fn="cross_entropy").result()

        for j, res in enumerate(fwd_result.loss_fn_outputs):
            logprobs = res["logprobs"].to_numpy()
            weights = datums[j].loss_fn_inputs["weights"].to_numpy()
            asst_logprobs = logprobs[weights > 0]
            loss = -asst_logprobs.mean() if len(asst_logprobs) > 0 else 0.0

            scored_ex = dict(batch[valid_indices[j]])
            scored_ex["difficulty_score"] = float(loss)
            scored.append(scored_ex)

    return scored


def main():
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=MODEL, rank=1)

    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Single pass through the full dataset, bucket by source
    print("Streaming Tulu dataset and collecting examples by source...")
    tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    buckets = {source: [] for source in SOURCES}

    for ex in tqdm(tulu, desc="Collecting"):
        source = ex.get("source")
        if source in buckets:
            buckets[source].append(ex)

    for source, examples in buckets.items():
        print(f"\n{source}: {len(examples)} examples")

    # Score each source and write output
    for source, examples in buckets.items():
        output_file = SOURCES[source]
        print(f"\nScoring source: {source}")
        print(f"  {len(examples)} examples -> {output_file}")

        if not examples:
            print("  No examples found, skipping.")
            continue

        scored = score_examples(examples, tc, renderer)

        with open(output_file, "w") as f:
            for item in scored:
                f.write(json.dumps(item) + "\n")

        print(f"  Saved {len(scored)} scored examples to {output_file}")

    print("\nDone.")


if __name__ == "__main__":
    main()
