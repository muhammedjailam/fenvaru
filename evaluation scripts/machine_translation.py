##################################
# Machine Translation Evaluation #
##################################
# Author: Mohamed Jailam
# Project: Dhivehi-Eval Benchmark Suite
# Description: Evaluation script for benchmarking Dhivehi↔English translation METEOR metrics.
#              Designed to support multiple AI models (via OpenRouter) and compute
#              reproducible translation evaluation results using custom test sets.
# Repository: https://github.com/muhammedjailam/fenvaru
# License: MIT License

# Install necessary packages
!pip install nltk pandas requests

# Import required libraries
import requests
import pandas as pd
import nltk
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score as base_meteor_score
from nltk.corpus.reader import wordnet as wn_reader
from nltk.corpus import wordnet
import os
import json

# Download default WordNet if not available
nltk.download('wordnet')

# Initialize OpenRouter API
OPENROUTER_API_KEY = "your-api-key"

# Define models you want to evaluate
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

# User-defined system prompt
SYSTEM_PROMPT = "You are a professional machine translator. Translate the following sentence to Dhivehi."

DATASET_PATH = "ml-translation.json"  # 🔁 Replace with your dataset path
CUSTOM_WORDNET_DIR = "dhivehi_wordnet/"  # e.g. "/content/my_wordnet"

# Load dataset from JSON file
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='nltk.corpus.reader.wordnet')

# Load custom WordNet if provided
if CUSTOM_WORDNET_DIR and os.path.isdir(CUSTOM_WORDNET_DIR):
    print(f"🔁 Loading custom WordNet from {CUSTOM_WORDNET_DIR}")
    wn_custom = wn_reader.WordNetCorpusReader(CUSTOM_WORDNET_DIR, None)
    wordnet._ensure_loaded = lambda: None  # disable internal loader
    wordnet.__dict__.update(wn_custom.__dict__)
else:
    print("✅ Using default WordNet")

# Function to call OpenRouter API
def query_openrouter(model, user_input, system_prompt):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://chat.openai.com",  # required
        "X-Title": "Dhivehi Benchmarking"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[ERROR] {model} | {e}")
        return ""

# Function to compute METEOR
def meteor_score(reference, prediction):
    return base_meteor_score([reference.split()], prediction.split())

# Dictionary to store results
results = {model: [] for model in MODELS_TO_TEST}

# Evaluate each model
for model in MODELS_TO_TEST:
    print(f"\n🔍 Evaluating model: {model}")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prediction = query_openrouter(model, row["english"], SYSTEM_PROMPT)
        score = meteor_score(row["dhivehi"], prediction)
        results[model].append(score)

# Save detailed per-sample results
detailed_df = pd.DataFrame({
    "enlish": df["english"],
    "dhivehi": df["dhivehi"],
    **{model: results[model] for model in MODELS_TO_TEST}
})
detailed_df.to_csv("detailed_results.csv", index=False)

# Leaderboard generation
leaderboard = pd.DataFrame([
    {"model": model, "avg_meteor": sum(scores) / len(scores)}
    for model, scores in results.items()
])
leaderboard = leaderboard.sort_values("avg_meteor", ascending=False)
leaderboard.to_csv("leaderboard.txt", index=False)

# Print leaderboard
print("\n📊 Leaderboard (Average METEOR Scores):\n")
print(leaderboard.to_string(index=False))

