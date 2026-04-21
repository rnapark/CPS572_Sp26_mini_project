"""
Train a model (minimal SFT), save checkpoint, and publish it.

TODO:
  - Replace DEMO_CONVERSATIONS with your task-specific training data
  - Tune hyperparameters (learning rate, batch size, number of steps, LoRA rank)
  - Add validation / early stopping as needed

Usage:
    python evaluation/train_and_publish.py
    python evaluation/train_and_publish.py --num_steps 20
    python evaluation/train_and_publish.py --no_publish   # skip publishing
"""

import argparse
import json
import subprocess
import os
import random
import math

import numpy as np
from datasets import load_dataset, load_from_disk

import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

#MODEL = "meta-llama/Llama-3.2-3B"
#MODEL = "meta-llama/Llama-3.2-1B"    # Smaller, faster for development
MODEL = "meta-llama/Llama-3.1-8B"    # Recommended for final submission

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

def load_tulu_sources(target_ratio_if=1.0, target_ratio_math=0.0):
    """Load all scored Tulu examples into separate per-source pools. Ratios are enforced at batch time."""
    source_configs = [
        ("tulu_source1_scored.jsonl", "source1", target_ratio_if > 0),
        ("tulu_source2_scored.jsonl", "source2", target_ratio_if > 0),
        ("tulu_source3_scored.jsonl", "source3", target_ratio_math > 0),
        ("tulu_source4_scored.jsonl", "source4", target_ratio_math > 0),
    ]

    pools = {}
    for filepath, key, should_load in source_configs:
        if not should_load:
            continue
        data = []
        with open(filepath) as f:
            for line in f:
                data.append(json.loads(line))

        # Z-score normalize within this source for cross-source comparability
        scores = np.array([ex["difficulty_score"] for ex in data])
        normalized = (scores - scores.mean()) / scores.std() if scores.std() > 0 else np.zeros_like(scores)
        for i, ex in enumerate(data):
            ex["_normalized_score"] = float(normalized[i])

        pools[key] = data
        print(f"  {filepath}: {len(data)} examples loaded")

    return pools

def filter_tulu_examples(tulu, tulu_samples, target_ratio_if=0.5, target_ratio_math=0.25):
    target_amount_if = int(tulu_samples * target_ratio_if)
    target_amount_math = int(tulu_samples * target_ratio_math)

    target_source1_if = "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980"
    target_source2_if = "ai2-adapt-dev/no_robots_converted"
    target_source3_math = "allenai/tulu-3-sft-personas-math-grade"
    target_source4_math = "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k"

    target_amount_if1 = 4500
    target_amount_if2 = target_amount_if - target_amount_if1

    tulu_target1 = []
    tulu_target2 = []
    tulu_target3 = []
    tulu_target4 = []
    tulu_other = []

    for ex in tulu:
        if ex.get("source") == target_source1_if:
            if len(tulu_target1) < target_amount_if1:
                tulu_target1.append(ex)
        elif ex.get("source") == target_source2_if:
            if len(tulu_target2) < target_amount_if2:
                tulu_target2.append(ex)
        elif ex.get("source") == target_source3_math:
            if len(tulu_target3) < target_amount_math//2:
                tulu_target3.append(ex)
        elif ex.get("source") == target_source4_math:
            if len(tulu_target4) < target_amount_math//2:
                tulu_target4.append(ex)
        else:
            if len(tulu_other) < (tulu_samples - target_amount_if - target_amount_math):
                tulu_other.append(ex)

        # stop early when full
        if len(tulu_target1) >= target_amount_if1 and len(tulu_target2) >= target_amount_if2 and len(tulu_target3) >= target_amount_math//2 and len(tulu_target4) >= target_amount_math//2 and len(tulu_other) >= (tulu_samples - target_amount_if - target_amount_math):
            break

    # backfill if target is too small
    if len(tulu_target1) < target_amount_if1:
        print(f"Warning: only found {len(tulu_target1)} target1 samples, backfilling...")
        needed = target_amount_if1 - len(tulu_target1)
        tulu_target1.extend(tulu_other[:needed])
        tulu_other = tulu_other[needed:]
    if len(tulu_target2) < target_amount_if2:
        print(f"Warning: only found {len(tulu_target2)} target2 samples, backfilling...")
        needed = target_amount_if2 - len(tulu_target2)
        tulu_target2.extend(tulu_other[:needed])
        tulu_other = tulu_other[needed:]
    if len(tulu_target3) < target_amount_math//2:
        print(f"Warning: only found {len(tulu_target3)} target3 samples, backfilling...")
        needed = target_amount_math - len(tulu_target3)
        tulu_target3.extend(tulu_other[:needed])
        tulu_other = tulu_other[needed:]
    if len(tulu_target4) < target_amount_math//2:
        print(f"Warning: only found {len(tulu_target4)} target4 samples, backfilling...")
        needed = target_amount_math - len(tulu_target4)
        tulu_target4.extend(tulu_other[:needed])
        tulu_other = tulu_other[needed:]

    tulu_subset = tulu_target1 + tulu_target2 + tulu_target3 + tulu_target4 + tulu_other
    random.shuffle(tulu_subset)

    print(f"Tulu subset: {len(tulu_subset)} total")
    print(f"  Target source 1: {len(tulu_target1)}")
    print(f"  Target source 2: {len(tulu_target2)}")
    print(f"  Target source 3: {len(tulu_target3)}")
    print(f"  Target source 4: {len(tulu_target4)}")
    print(f"  Other: {len(tulu_other)}")
    return tulu_subset

def scoring_function(ifeval, gsm8k, humaneval, split=0.7):
    # cap scores at reasonable thresholds and average
    # forces model to improve weakest task
    #factor allows us to push model to higher performance on all tasks, but still requires balance
    base = (
        ifeval +
        gsm8k +
        humaneval
    ) / 3

    # penalize imbalance
    min_task = min(
        (ifeval) / .5,
        (gsm8k) /.45,
        (humaneval) / .3
    )

    return split * base + (1 - split) * min_task

def load_scored_data(filename, temperature=2.0):
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    scores = np.array([ex["difficulty_score"] for ex in data])
    exp_scores = np.exp((scores - np.max(scores)) / temperature)
    probs = exp_scores / exp_scores.sum()
    
    for i, ex in enumerate(data):
        ex["sampling_prob"] = probs[i]
    return data

def weighted_sample(dataset, count):
    if count == 0 or not dataset: return []
    probs = [ex["sampling_prob"] for ex in dataset]
    indices = np.random.choice(len(dataset), size=count, replace=False, p=probs)
    return [dataset[i] for i in indices]

def weighted_sample_dynamic(examples, count, temperature):
    """Sample from a pool using difficulty scores at the given temperature."""
    if count == 0 or not examples: return []
    count = min(count, len(examples))
    scores = np.array([ex["_normalized_score"] for ex in examples])
    exp_scores = np.exp((scores - np.max(scores)) / temperature)
    probs = exp_scores / exp_scores.sum()
    indices = np.random.choice(len(examples), size=count, replace=False, p=probs)
    return [examples[i] for i in indices]

def get_tulu_temp(step, total_steps, temp_start=4.0, temp_end=1.0):
    """Cosine schedule: start warm (diverse/easy) and cool to hard examples over training."""
    progress = step / total_steps
    return temp_end + (temp_start - temp_end) * 0.5 * (1 + math.cos(math.pi * progress))

def get_lr(step, total_steps, base_lr, warmup_steps=100):
    # cosine decay with linear warmup
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def build_batch(gsm8k_data, tulu_pools, tulu_ratio_if, tulu_ratio_math, opencode_data, step, total_steps, batch_size=4):
    """
    Returns a batch of examples. Tulu is sampled per-source with difficulty temperature schedule.
    """
    progress = step / total_steps

    # GSM8K phase-based schedule
    if progress < 0.3:
        target_ratio = 0.2
    elif progress < 0.7:
        target_ratio = 0.2
    else:
        target_ratio = 0.2

    gsm8k_count = max(1, np.random.binomial(batch_size, target_ratio))
    remaining = batch_size - gsm8k_count
    opencode_count = max(1, remaining // 8)
    tulu_count = max(0, remaining - opencode_count)

    # Difficulty temperature: warm (diverse) at start, cool (harder) at end
    tulu_temp = get_tulu_temp(step, total_steps)

    # Allocate tulu budget across sources by ratio, sample each source independently
    if_count = round(tulu_count * tulu_ratio_if)
    math_count = tulu_count - if_count
    if1_count = if_count // 2
    if2_count = if_count - if1_count
    math3_count = math_count // 2
    math4_count = math_count - math3_count

    tulu_samples = []
    for key, count in [("source1", if1_count), ("source2", if2_count),
                       ("source3", math3_count), ("source4", math4_count)]:
        pool = tulu_pools.get(key, [])
        sampled = weighted_sample_dynamic(pool, count, tulu_temp)
        tulu_samples.extend([dict(ex, _source="tulu") for ex in sampled])

    # Safety for gsm8k and opencode
    gsm8k_count = min(gsm8k_count, len(gsm8k_data))
    opencode_count = min(opencode_count, len(opencode_data))

    gsm8k_samples = weighted_sample(gsm8k_data, gsm8k_count)
    opencode_samples = [dict(ex, _source="opencode") for ex in random.sample(opencode_data, opencode_count)]

    for ex in gsm8k_samples:
        ex["_source"] = "gsm8k"

    batch = gsm8k_samples + tulu_samples + opencode_samples
    random.shuffle(batch)
    return batch

def example_to_convo(example):
    """
    Convert an example to a conversation format [{'role': ..., 'content': ...}]
    Supports:
      - GSM8K style: {"question": ..., "answer": ...}
      - Chat style: {"messages": [...]}
    Returns None if no valid convo can be created.
    """
    source = example.get("_source", None)
    if "question" in example and "answer" in example: # GSM8K style
        base_answer = example["answer"]
        enhanced_answer = f"Let's think step by step.\n{base_answer}"
        convo = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": enhanced_answer},
        ]
        return convo, source

    if "messages" in example: # Tulu
        messages = example["messages"]
        pairs = [
            (messages[i]["content"], messages[i+1]["content"])
            for i in range(len(messages)-1)
            if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant"
        ]
        if pairs:
            # use a mix of longest and last pair
            if random.random() < 0.3:
                question, answer = max(
                    pairs,
                    key=lambda qa: len(qa[1].split())  # word-based length is more stable
                )
            else:
                # last 70%, just take last back-and-forth pair
                question, answer = pairs[-1]
            convo = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            return convo, source
    
    if "input" in example: # OpenCodeInstruct
        convo = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return convo, source

    return None, None  # unsupported format

def token_too_long_gsm8k(convo, renderer):
    datum1 = conversation_to_datum(
        convo,
        renderer,
        max_length=512,
        train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    
    datum2 = conversation_to_datum(
        convo,
        renderer,
        max_length=513,
        train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
    )
    return datum1!=datum2


def filter_gsm8k_examples(examples_array, renderer):
    good_examples = []
    for example in examples_array:
        convo, _ = example_to_convo(example)
        if convo is None:
            continue
        if not token_too_long_gsm8k(convo, renderer):
            good_examples.append(example)
    print(f"Filtered examples: {len(examples_array)} -> {len(good_examples)} (max token length 512)")
    return good_examples



def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--eval_limit", type=int, default=50, help="Number of examples for intermediate evaluation")
    parser.add_argument("--checkpoint_name", type=str, default="demo", help="Checkpoint name")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Setup
    print(f"Model: {MODEL}")
    print(f"Training for {args.num_steps} steps with batch size {args.batch_size}, learning rate {args.lr}, LoRA rank {args.rank}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Prepare training data
    print("Loading 3 datasets...")

    # Load each dataset (only the train split)
    # The datasets are large, so stream and take random subset
    # this won't be a truly random subset but good enough for now
    gsm8k_subset = load_scored_data("gsm8k_scored.jsonl", temperature=2.0)
    #opencode = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True).shuffle(seed=42)
    opencode = load_from_disk("opencode_filtered").shuffle(seed=42)

    tulu_ratio_if = 0.6
    tulu_ratio_math = 0.4
    opencode_samples = 10000
    tulu_pools = load_tulu_sources(target_ratio_if=tulu_ratio_if, target_ratio_math=tulu_ratio_math)

    # Take the first ___ random samples from the streamed dataset
    # Change ratios later, or change to selective sampling
    #tulu_examples = [example for _, example in zip(range(tulu_samples), tulu)]
    #opencode_examples = [example for _, example in zip(range(opencode_samples), opencode)]
    opencode_subset = random.sample(list(opencode), min(opencode_samples, len(opencode)))


    # Create training client
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
    print("  Training client ready")

    # Train
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")

    best_score = -1
    best_checkpoint_path = None
    patience = 20
    min_delta = 0.0
    steps_since_improve = 0

    for step in range(args.num_steps):
        raw_batch = build_batch(gsm8k_subset, tulu_pools, tulu_ratio_if, tulu_ratio_math, opencode_subset, step=step, total_steps=args.num_steps, batch_size=args.batch_size)
        batch = []
        sources = []
        for example in raw_batch:
            convo, source = example_to_convo(example)
            if convo:
                datum = conversation_to_datum(
                    convo,
                    renderer,
                    max_length=512,
                    train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
                )
                batch.append(datum)
                sources.append(source)
        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")

        # added learning rate scheduler with cosine decay and linear warmup
        lr = get_lr(step, args.num_steps, args.lr)

        adam_params = types.AdamParams(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8
        )

        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Compute loss
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])

        weights_list = [d.loss_fn_inputs["weights"].to_numpy() for d in batch]  # convert TensorData
        weights = np.concatenate(weights_list)

        # Expand token-level sources
        token_sources = np.concatenate([
            np.full(len(w), src, dtype="U10")
            for w, src in zip(weights_list, sources)
        ])

        gsm8k_boost = 1.3
        weights = weights.copy()
        # only weight asst tokens 
        gsm_mask = (token_sources == "gsm8k") & (weights > 0)
        weights[gsm_mask] *= gsm8k_boost
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        if(step % 50 == 0 or step == args.num_steps - 1):
            print(f"  Step {step+1}/{args.num_steps} | Loss: {loss:.4f}")
        
        # Save checkpoint + evaluate every 100 steps
        if (step + 1) % 100 == 0 or step == args.num_steps - 1:
            checkpoint_name = f"{args.checkpoint_name}_step{step+1}"
            print(f"\nSaving intermediate checkpoint '{checkpoint_name}'...")
            ckpt = tc.save_weights_for_sampler(name=checkpoint_name).result()
            checkpoint_path = ckpt.path
            print(f"  Checkpoint saved: {checkpoint_path}")

            # Run evaluation on this checkpoint and capture stdout
            cmd = [
                "python", "-m", "evaluation.eval_all",
                "--checkpoint_path", checkpoint_path,
                "--base_model", MODEL,
                "--limit", str(args.eval_limit)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            stdout = result.stdout

            # Parse JSON block from stdout
            try:
                start = stdout.index("{")
                end = stdout.rindex("}") + 1
                eval_metrics = json.loads(stdout[start:end])
                print(f"Evaluation metrics: {eval_metrics}")
            except ValueError:
                print("Could not parse evaluation output")
                eval_metrics = {}

            # Track best checkpoint
            current_ifeval = eval_metrics.get("google/IFEval/final_acc", 0.0)
            current_gsm8k = eval_metrics.get("openai/gsm8k/accuracy", 0.0)
            current_humaneval = eval_metrics.get("openai/openai_humaneval/accuracy", 0.0)
            current_score = scoring_function(current_ifeval, current_gsm8k, current_humaneval)
            if current_score > best_score + min_delta:
                best_score = current_score
                best_checkpoint_path = checkpoint_path
                steps_since_improve = 0
                print(f"New best checkpoint at step {step+1} with score {best_score:.4f}")

            else:
                print(f"Not a new best checkpoint, score {current_score:.4f}")
                steps_since_improve += 1

            if steps_since_improve >= patience:
                print(f"No improvement for {patience} evaluations, stopping early at step {step+1}")
                break

    # Save checkpoint, best intm if there is one
    print(f"\nSelecting best checkpoint...")
    if best_checkpoint_path:
        checkpoint_path = best_checkpoint_path
        print(f"Using best checkpoint: {checkpoint_path} (score={best_score:.4f})")
    else:
        print("No best checkpoint found, saving final state...")
        ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
        checkpoint_path = ckpt.path
        print(f"Checkpoint saved: {checkpoint_path}")

    # Publish
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    # Save checkpoint info
    info = {
        "checkpoint_path": checkpoint_path,
        "base_model": MODEL,
        "renderer_name": renderer_name,
        "training": {
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "lora_rank": args.rank,
        },
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f"  python -m evaluation.eval_all --checkpoint_path \"{checkpoint_path}\" --base_model {MODEL}")


if __name__ == "__main__":
    main()
