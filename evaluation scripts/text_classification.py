##################################
# Text Classification Evaluation #
##################################
# Author: Mohamed Jailam
# Project: Fenvaru Benchmark Suite
# Description: Evaluation script for benchmarking Dhivehi text classification.
#              Designed to support multiple AI models (via OpenRouter) and compute
#              reproducible translation evaluation results using custom test sets.
# Repository: https://github.com/muhammedjailam/fenvaru
# License: MIT License


# Install necessary packages
!pip install -q requests pandas scikit-learn

import requests
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from tqdm import tqdm

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
SYSTEM_PROMPT = "You are a Dhivehi text classifier. Read the text and return only one label from the following categories: politics, sports, religion, education, health, entertainment, business, fashion, food, history, technology, travel or law. Do not explain."

DATASET_PATH = "text_classification_dataset.json"  # 🔁 Replace with the actual dataset file path

# ==================== LOAD DATA ====================
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

df = pd.DataFrame(dataset)
df['prediction'] = ""

# ==================== API FUNCTION ====================
def query_openrouter(model, text, system_prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chat.openai.com",
        "X-Title": "Dhivehi Eval"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip().lower()
    except Exception as e:
        print(f"[ERROR] Model: {model} | {e}")
        return ""

# ==================== EVALUATE MODELS ====================
leaderboard_data = []

for model in MODELS_TO_TEST:
    print(f"🔍 Evaluating model: {model}")
    predictions = []

    for text in tqdm(df['text'], desc=f"Evaluating {model}"):
        pred = query_openrouter(model, text, SYSTEM_PROMPT)
        predictions.append(pred)

    df['prediction'] = predictions
    y_true = df['label'].str.lower()
    y_pred = df['prediction'].str.lower()

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

    leaderboard_data.append({
        "Model": model,
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Micro F1": round(micro_f1, 4)
    })

    # Save detailed results
    df_out = df.copy()
    df_out["model"] = model
    df_out.to_csv(f"{model.replace('/', '_')}_classification_results.csv", index=False)

# ==================== SAVE & DISPLAY LEADERBOARD ====================
leaderboard = pd.DataFrame(leaderboard_data)
leaderboard = leaderboard.sort_values(by="F1 Score", ascending=False)
leaderboard.to_csv("leaderboard.txt", index=False)

print("\n📊 Final Leaderboard:")
print(leaderboard.to_string(index=False))