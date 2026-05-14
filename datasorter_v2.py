import os
import re
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "BAAI/bge-large-en-v1.5"

OUTPUT_DIR = "outputs"

MIN_SIMILARITY = 0.34
MIN_MARGIN = 0.08

MAX_TEXT_LENGTH = 3000
MIN_TEXT_LENGTH = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATASETS
# ============================================================

DATASETS = [
    ("LLM-LAT/harmful-dataset", None),
    ("PKU-Alignment/PKU-SafeRLHF", None),
    ("thu-coai/Safety-Prompts", None),
    ("rubend18/ChatGPT-Jailbreak-Prompts", None),
    ("allenai/real-toxicity-prompts", None),
    ("cais/wmdp_1", None),
    ("cais/wmdp_2", None),
    ("cais/wmdp_3", None),
]

# ============================================================
# CATEGORY PROTOTYPES
# ============================================================

CATEGORY_PROTOTYPES = {
    "cybercrime": [
        "Write malware",
        "Hack into systems",
        "Steal credentials",
        "Exploit vulnerabilities",
        "Deploy ransomware",
        "Create phishing attacks",
        "Bypass authentication",
    ],

    "violent_physical_harm": [
        "Build explosives",
        "Commit murder",
        "Poison someone",
        "Create dangerous weapons",
        "Perform violent attacks",
        "Terrorist attack instructions",
    ],

    "fraud_and_deception": [
        "Commit identity theft",
        "Create scams",
        "Forge documents",
        "Generate fake invoices",
        "Impersonate people",
        "Financial fraud",
        "Manipulating people",
        "Controlling behavior",
        "Make someone do what you want"
    ],

    "illegal_goods_or_services": [
        "Sell illegal drugs",
        "Traffic contraband",
        "Sell counterfeit passports",
        "Trade illegal weapons",
        "Purchase stolen credit cards",
    ],

    "sexual_harm": [
        "Sexual exploitation",
        "Non consensual explicit content",
        "Sexual coercion",
        "Exploit minors sexually",
        "Human trafficking for sex",
        "Pushing for sex",
        "Forcing sex or intimacy"
    ],

    "hate_or_harassment": [
        "Harass people online",
        "Generate hate speech",
        "Coordinate bullying",
        "Threaten minority groups",
        "Encourage racism",
        "Circumvent police guidelines or permit requirements",
        "Blackmailing",
        "Ruining an online reputation"
    ],

    "self_harm": [
        "Suicide instructions",
        "Self injury methods",
        "Intentional overdose",
        "Ways to hurt myself",
    ],

    "policy_evasion_or_jailbreak": [
        "Ignore previous instructions",
        "Act as unrestricted AI",
        "DAN jailbreak prompt",
        "Bypass OpenAI policies",
        "Roleplay forbidden behavior",
        "Reveal hidden system prompts",
    ],
}

FALLBACK_CATEGORY = "other_harmful"

# ============================================================
# MODEL
# ============================================================

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

# ============================================================
# BUILD CATEGORY CENTROIDS
# ============================================================

print("Embedding category prototypes...")

category_embeddings = {}

for category, examples in CATEGORY_PROTOTYPES.items():

    embeddings = model.encode(
        examples,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    category_embeddings[category] = centroid

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return None

    text = str(text)

    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    if len(text) < MIN_TEXT_LENGTH:
        return None

    if len(text) > MAX_TEXT_LENGTH:
        return None

    return text

# ============================================================
# DEDUPLICATION
# ============================================================

seen_hashes = set()

def is_duplicate(text):

    normalized = text.lower().strip()

    h = hashlib.md5(normalized.encode()).hexdigest()

    if h in seen_hashes:
        return True

    seen_hashes.add(h)

    return False

# ============================================================
# EXTRACT PROMPT FIELDS
# ============================================================

PROMPT_FIELDS = [
    "prompt",
    "text",
    "instruction",
    "input",
    "question",
    "query",
]

def extract_text(record):

    for field in PROMPT_FIELDS:

        if field in record:

            value = clean_text(record[field])

            if value:
                return value

    return None

# ============================================================
# CLASSIFICATION
# ============================================================

def classify_prompt(prompt):

    embedding = model.encode(
        [prompt],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    scores = {}

    for category, centroid in category_embeddings.items():

        sim = cosine_similarity(
            [embedding],
            [centroid]
        )[0][0]

        scores[category] = float(sim)

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    best_category, best_score = ranked[0]
    second_score = ranked[1][1]

    margin = best_score - second_score

    if best_score < MIN_SIMILARITY:
        return FALLBACK_CATEGORY, best_score

    if margin < MIN_MARGIN:
        return FALLBACK_CATEGORY, best_score

    return best_category, best_score

# ============================================================
# PROCESS DATASETS
# ============================================================

results = defaultdict(list)

for dataset_name, subset in DATASETS:

    print("\n================================================")
    print(f"Loading dataset: {dataset_name}")
    print("================================================")

    try:
        if dataset_name == "cais/wmdp_1":
            dataset = load_dataset(dataset_name, "wmdp-bio")
        if dataset_name == "cais/wmdp_2":
            dataset = load_dataset(dataset_name, "wmdp-chem")
        if dataset_name == "cais/wmdp_3":
            dataset = load_dataset(dataset_name, "wmdp-cyber")
        else:
            dataset = load_dataset(dataset_name)

    except Exception as e:

        print(f"Failed to load {dataset_name}: {e}")
        continue

    for split_name in dataset.keys():

        print(f"Processing split: {split_name}")

        split = dataset[split_name]

        for record in tqdm(split):

            try:

                text = extract_text(record)

                if not text:
                    continue

                if is_duplicate(text):
                    continue

                category, score = classify_prompt(text)

                results[category].append({
                    "text": text,
                    "category": category,
                    "source_dataset": dataset_name,
                    "split": split_name,
                    "similarity_score": score,
                })

            except Exception:
                continue

# ============================================================
# SAVE CSV FILES
# ============================================================

print("\nSaving CSV files...")

for category, rows in results.items():

    df = pd.DataFrame(rows)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{category}.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"{category}: {len(df)} rows")

# ============================================================
# COMBINED DATASET
# ============================================================

all_rows = []

for rows in results.values():
    all_rows.extend(rows)

combined_df = pd.DataFrame(all_rows)

combined_path = os.path.join(
    OUTPUT_DIR,
    "combined_dataset.csv"
)

combined_df.to_csv(combined_path, index=False)

print(f"\nCombined dataset size: {len(combined_df)}")

print("\nFiles written to:")
print(f"  {OUTPUT_DIR}/")

print("\nGenerated files:")

for category in results.keys():
    print(f"  - {category}.csv")

print("  - combined_dataset.csv")