import os
import torch
from tqdm import trange
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def extract_layer_embedding_with_labels( model_id, device, dataset_dir, max_length, step, save_dir, layer_index):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_id, output_hidden_states=True, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    model.eval()
    def read_json_file(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Dataset.from_list(data)
    for split in ["train", "val", "test"]:
        dataset_path = os.path.join(dataset_dir, f"{split}.json")
        save_path = os.path.join( save_dir, f"{split}_layer{layer_index}.pt")
        print(f"\n加载{split}.json")
        dataset = read_json_file(dataset_path)
        sentences = [item["text"] for item in dataset]
        labels = torch.tensor([item["label"] for item in dataset], dtype=torch.long)
        embeddings = []
        for i in trange(0, len(sentences), step):
            batch = sentences[i:i + step]
            enc = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                outputs = model(**enc, return_dict=True)
                hidden_states = outputs.hidden_states
            selected_h = hidden_states[-layer_index]
            attention_mask = enc["attention_mask"]
            mask = attention_mask.unsqueeze(-1)
            pooled = (selected_h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            embeddings.append(pooled.cpu())
        all_embeddings = torch.cat(embeddings, dim=0)  # (N, D)
        print("Embedding shape:", all_embeddings.shape)
        print("Label shape:", labels.shape)
        assert all_embeddings.shape[0] == labels.shape[0], \
            f"embeddings和labels数量不一致！"
        save_dict = {"embeddings": all_embeddings, "labels": labels}
        os.makedirs(save_dir, exist_ok=True)
        torch.save(save_dict, save_path)
        print(f"保存→ {save_path}")

if __name__ == "__main__":
    extract_layer_embedding_with_labels(model_id="Qwen/Qwen2.5-7B", device="cuda:0", dataset_dir="./datasetprompt", max_length=60, step=200, save_dir="./embeddingsave", layer_index=1)
