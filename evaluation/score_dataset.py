import json
import numpy as np
import tinker
from datasets import load_dataset
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm import tqdm

MODEL = "meta-llama/Llama-3.2-3B"
OUTPUT_FILE = "gsm8k_scored.jsonl"
BATCH_SIZE = 8 

def main():
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=MODEL, rank=1) 
    
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    print("Loading GSM8K...")
    dataset = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(7473))
    
    scored_data = []
    
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Scoring with Tinker"):
        batch_examples = dataset.select(range(i, min(i + BATCH_SIZE, len(dataset))))
        batch_datums = []
        valid_indices = []

        for idx, ex in enumerate(batch_examples):

            convo = [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": f"Let's think step by step.\n{ex['answer']}"},
            ]
            
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=512,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )
            batch_datums.append(datum)
            valid_indices.append(idx)

        fwd_result = tc.forward_backward(batch_datums, loss_fn="cross_entropy").result()

        for j, res in enumerate(fwd_result.loss_fn_outputs):
            logprobs = res["logprobs"].to_numpy()
            weights = batch_datums[j].loss_fn_inputs["weights"].to_numpy()
            
            asst_logprobs = logprobs[weights > 0]
            
            loss = -asst_logprobs.mean() if len(asst_logprobs) > 0 else 0.0
            
            original_ex = dict(batch_examples[valid_indices[j]])
            original_ex["difficulty_score"] = float(loss)
            scored_data.append(original_ex)

    # Save to local JSONL
    with open(OUTPUT_FILE, "w") as f:
        for item in scored_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Done! Scored data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()