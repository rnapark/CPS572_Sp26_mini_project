"""
Train a model (minimal SFT), save checkpoint, and publish it.

NOTE: This is a TOY EXAMPLE that trains for a few steps on dummy data
to verify the full workflow end-to-end. You should replace the training
data and training logic with your own implementation.

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
import os
import random

import numpy as np
from datasets import load_dataset
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "meta-llama/Llama-3.2-3B"
#MODEL = "meta-llama/Llama-3.2-1B"    # Smaller, faster for development
# MODEL = "meta-llama/Llama-3.1-8B"    # Recommended for final submission

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# TODO: TOY DATA, replace with your own training data
DEMO_CONVERSATIONS = [
    [
        {"role": "user", "content": "What is 15 + 27?"},
        {"role": "assistant", "content": "15 + 27 = 42"},
    ],
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
    [
        {"role": "user", "content": "Write a Python function that returns the sum of two numbers."},
        {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
    ],
    [
        {"role": "user", "content": "What is 8 * 7?"},
        {"role": "assistant", "content": "8 * 7 = 56"},
    ],
    [
        {"role": "user", "content": "Translate 'hello' to Spanish."},
        {"role": "assistant", "content": "Hola"},
    ],
    [
        {"role": "user", "content": "What is the square root of 144?"},
        {"role": "assistant", "content": "The square root of 144 is 12."},
    ],
    [
        {"role": "user", "content": "Write a Python function to check if a number is even."},
        {"role": "assistant", "content": "def is_even(n):\n    return n % 2 == 0"},
    ],
    [
        {"role": "user", "content": "List the first 5 prime numbers."},
        {"role": "assistant", "content": "The first 5 prime numbers are: 2, 3, 5, 7, 11."},
    ],
]


def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--checkpoint_name", type=str, default="demo", help="Checkpoint name")
    parser.add_argument("--no_publish", action="store_true", help="Skip publishing")
    args = parser.parse_args()

    # Setup
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Prepare training data
    print("Loading 3 datasets...")

    # Load each dataset (only the train split)
    # The datasets are large, so stream and take random subset
    # this won't be a truly random subset but good enough for now
    gsm8k = load_dataset("openai/gsm8k", "main", split="train", streaming=True).shuffle(seed=42)
    tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True).shuffle(seed=42)
    #opencodeinstruct = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True).shuffle(seed=42)

    # Take the first 1000 random samples from the streamed dataset
    # Change ratios later, or change to selective sampling
    gsm8k_subset = [example for _, example in zip(range(1000), gsm8k)]
    tulu_subset = [example for _, example in zip(range(5000), tulu)]
    #opencodeinstruct_subset = [example for _, example in zip(range(100), opencodeinstruct)]
    # interleave so each batch has a mix of datasets

    interleaved = []
    t_idx = 0
    for i, g_example in enumerate(gsm8k_subset):
        interleaved.append(g_example)
        # Every 2 GSM8K examples, add 1 Tulu example
        if (i + 1) % 2 == 0 and t_idx < len(tulu_subset):
            interleaved.append(tulu_subset[t_idx])
            t_idx += 1
    print("Preparing training data...")
    

    all_data = []

    for example in interleaved:
        if "question" in example and "answer" in example:
            # gsm8k format
            question = example["question"]
            #answer = example["answer"]
            answer = example["answer"].strip()
            #print(f"User: {question}\nAssistant: {answer}\n")
        elif "messages" in example:
            # tulu or opencodeinstruct format
            question = None
            answer = None
            # take LAST assistant message paired with preceding user
            messages = example["messages"]

            for i in range(len(messages)-1):
                if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                    question = messages[i]["content"]
                    answer = messages[i+1]["content"]

        convo = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

        datum = conversation_to_datum(
            convo,
            renderer,
            max_length=512,
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        all_data.append(datum)

    print(f"{len(all_data)} training examples prepared") 

    # Create training client
    print(f"Creating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
    print("  Training client ready")

    # Train
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")

    for step in range(args.num_steps):
        # Cycle through data
        start = (step * args.batch_size) % len(all_data)
        batch = [all_data[i % len(all_data)] for i in range(start, start + args.batch_size)]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        # Compute loss
        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        if(step % 10 == 0 or step == args.num_steps - 1):
            print(f"  Step {step+1}/{args.num_steps} | Loss: {loss:.4f}")

    # Save checkpoint
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

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
