# ==================== 
# QA Evaluation #
# ==================== 
# Author: Mohamed Jailam
# Project: Fenvaru Benchmark Suite
# Description: Evaluation script for benchmarking Dhivehi QA.
#              Designed to support multiple AI models (via OpenRouter) and compute
#              reproducible QA results using custom test sets.
# Repository: https://github.com/muhammedjailam/fenvaru
# License: MIT License

# Install necessary packages
!pip install -q requests pandas scikit-learn

import requests
import pandas as pd
import json
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# ==================== CONFIGURATION ====================
OPENROUTER_API_KEY = "your-api-key"  # 🔁 Replace with your actual API key
MODELS_TO_TEST = [
    "moonshotai/kimi-k2",
    "x-ai/grok-4",
    "x-ai/grok-3",
    "google/gemma-3n-e2b-it:free",
    "google/gemini-2.5-pro",
    "google/gemini-pro-1.5",
    "openai/o3-pro",
    "openai/gpt-4.1",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-chat-v3-0324",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.7-sonnet",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-3-70b",
    "mistralai/devstral-medium",
    "mistralai/mistral-large-2411",
    "qwen/qwen3-30b-a3b",
    "qwen/qwen-110b-chat",
    "amazon/nova-pro-v1",
    "amazon/nova-lite-v1",
]
SYSTEM_PROMPT = "You are a Dhivehi question answering system. Given a context and a question, extract the answer text from the context as precisely as possible. Only return the answer text, no explanation."

DATASET_PATH = "qa_dataset.json"  # 🔁 Replace with your dataset file

# ==================== LOAD DATA ====================
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

df = pd.DataFrame(dataset)

# Thaana dotted to non-dotted conversion
dotted_map = {
    'ޙ': 'ހ', 'ޚ': 'ހ', 'ޢ': 'އ', 'ޣ': 'އ',
    'ޞ': 'ސ', 'ޠ': 'ތ', 'ޡ': 'ތ', 'ޘ': 'ތ',
    'ޛ': 'ދ', 'ޝ': 'ސ', 'ޥ': 'ވ', 'ޤ': 'ގ'
}
def normalize(text):
    return ''.join([dotted_map.get(char, char) for char in text.strip()])

# ==================== API FUNCTION ====================
def query_openrouter(model, context, question, system_prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "Dhivehi QA Eval"
    }
    prompt = f"Context: {context}\nQuestion: {question}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[ERROR] Model: {model} | {e}")
        return ""

# ==================== METRIC FUNCTIONS ====================
def exact_match(prediction, ground_truth):
    return normalize(prediction) == normalize(ground_truth)

def token_f1(prediction, ground_truth):
    pred_tokens = normalize(prediction).split()
    truth_tokens = normalize(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not pred_tokens or not truth_tokens:
        return 0, 0, 0
    if not common:
        return 0, 0, 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

# ==================== EVALUATION ====================
leaderboard_data = []

for model in MODELS_TO_TEST:
    print(f"🔍 Evaluating model: {model}")
    f1s, ems, precisions, recalls = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {model}"):
        pred = query_openrouter(model, row["context"], row["question"], SYSTEM_PROMPT)
        gt = row["answer_text"]
        ems.append(int(exact_match(pred, gt)))
        p, r, f1 = token_f1(pred, gt)
        f1s.append(f1)
        precisions.append(p)
        recalls.append(r)

    leaderboard_data.append({
        "Model": model,
        "Precision": round(sum(precisions)/len(precisions), 4),
        "Recall": round(sum(recalls)/len(recalls), 4),
        "F1 Score": round(sum(f1s)/len(f1s), 4),
        "Exact Match": round(sum(ems)/len(ems), 4)
    })

# ==================== SAVE & DISPLAY LEADERBOARD ====================
leaderboard = pd.DataFrame(leaderboard_data)
leaderboard = leaderboard.sort_values(by="F1 Score", ascending=False)
leaderboard.to_csv("qa_leaderboard.txt", index=False)

print("\n📊 Final QA Leaderboard:")
print(leaderboard.to_string(index=False))