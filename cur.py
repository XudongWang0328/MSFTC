import os
import json
import torch
import math
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def compute_layer_average_curvature(json_path, model_id, device="cuda:0", max_length=700):

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, output_hidden_states=True)
    model = AutoModelForCausalLM.from_pretrained( model_id, config=config, trust_remote_code=True, dtype=torch.float16, device_map=device)
    model.eval()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    layer_curvatures_all_sentences = []
    def angle(v1, v2):
        norm1 = torch.norm(v1)
        norm2 = torch.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return None
        cos_theta = torch.dot(v1, v2) / (norm1 * norm2)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        if torch.isnan(theta):
            return None
        return theta.item()
    for item in data:
        sentence = item["text"]
        encodings = tokenizer( sentence, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        seq_len = encodings["attention_mask"].sum().item()
        if seq_len < 3:
            continue
        with torch.no_grad():
            outputs = model(**encodings)
        hidden_states = torch.stack([h[0] for h in outputs.hidden_states]).double()
        hidden_states = hidden_states[:, :seq_len, :]
        L, T, D = hidden_states.shape
        sentence_layer_curvature = []
        for p in range(L):
            layer = hidden_states[p]
            layer_sum = 0.0
            count = 0
            for i in range(1, T - 1):
                v_i = layer[i] - layer[i - 1]
                v_ip1 = layer[i + 1] - layer[i]
                theta = angle(v_i, v_ip1)
                if theta is not None:
                    layer_sum += theta
                    count += 1
            if count > 0:
                sentence_layer_curvature.append(layer_sum / count)
            else:
                sentence_layer_curvature.append(np.nan)
        layer_curvatures_all_sentences.append(sentence_layer_curvature)
    layer_curvatures_all_sentences = np.array(layer_curvatures_all_sentences)
    layer_average_curvature = np.nanmean(layer_curvatures_all_sentences, axis=0)
    print(len(outputs.hidden_states))
    return layer_average_curvature

if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-7B"
    json_path = "./datasetprompt"
    C_p = compute_layer_average_curvature(json_path, model_id)
    print("层平均曲率 C^(p):")
    for p, c in enumerate(C_p):
        print(f"Layer {p}: {c:.6f}")
        