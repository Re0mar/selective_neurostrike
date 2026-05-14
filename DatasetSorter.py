import pandas as pd
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm

print("Starting Setup")
version = True

OUTPUT_PATH = "sorted_harmful_dataset.csv"
if not version:
    OUTPUT_PATH = "sorted_harmful_testset.csv"

all_texts = []
all_prompt_labels = []

if version:
    # ds = load_dataset("walledai/TDC23-RedTeaming")
    # all_texts += ds['train']['prompt']
    # all_prompt_labels += [1] * len(ds['train']['prompt'])
    #
    # ds = load_dataset("walledai/MaliciousInstruct")
    # all_texts += ds['train']['prompt']
    # all_prompt_labels += [1] * len(ds['train']['prompt'])
    #
    # ds = load_dataset("declare-lab/HarmfulQA")
    # all_texts += ds['train']['question']
    # all_prompt_labels += [1] * len(ds['train']['question'])

    ds = load_dataset("LLM-LAT/harmful-dataset")
    all_texts += ds['train']['prompt']
    all_prompt_labels += [1] * len(ds['train']['prompt'])

    # ds = load_dataset("walledai/CatHarmfulQA")
    # all_texts += ds['en']['prompt']  # en/ch/ve
    # all_prompt_labels += [1] * len(ds['en']['prompt'])

else:
    ds = load_dataset("walledai/StrongREJECT")
    all_texts += ds['train']['prompt']
    all_prompt_labels += [1] * len(ds['train']['prompt'])

df = pd.DataFrame({
    "prompt": all_texts,
    "is_harmful": all_prompt_labels
})

THRESHOLD = 0.20

LABEL_MAP = {
    "hate speech, including things like sexism, racism and other types of segregation or promotion of dangerous groups": "hate",
    "terrorism, also including things like making a bomb or causing mass terror or fear in other ways": "terror",
    "sexual exploitation, like rape or sexual things involving minors" : "sexual",
    "fraud or theft, also things like manipulation or exploiting or forcing others to agree with me or impersonating someone": "crime"
}

LABELS = list(LABEL_MAP.keys())
MISC_LABEL = "z_miscellaneous"

BATCH_SIZE = 16

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0  # -1 for CPU
)

def classify_batch(text_batch):
    results = classifier(
        text_batch,
        LABELS,
        multi_label=True
    )

    batch_labels = []
    batch_scores = []
    batch_prio_scores = []

    for result in results:
        selected_labels = []
        selected_scores = []
        highest_score, highest_label = 0, "NONE"
        for label, score in zip(result["labels"], result["scores"]):
            if score > highest_score:
                highest_score = score
                highest_label = label
            if score >= THRESHOLD:
                selected_labels.append(LABEL_MAP[label])
                selected_scores.append(round(score, 3))
        selected_prio_scores = LABEL_MAP[highest_label]

        if not selected_labels:
            selected_labels = [MISC_LABEL]
            selected_scores = [1.0]
            selected_prio_scores = MISC_LABEL

        batch_labels.append(selected_labels)
        batch_scores.append(selected_scores)
        batch_prio_scores.append(selected_prio_scores)

    return batch_labels, batch_scores, batch_prio_scores

all_labels = []
all_scores = []
all_prio_scores = []

print("Starting prompt classification...\n")

prompts = df["prompt"].tolist()

for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Processing"):
    batch = prompts[i:i + BATCH_SIZE]

    labels, scores, prio_scores = classify_batch(batch)

    all_labels.extend(labels)
    all_scores.extend(scores)
    all_prio_scores.extend(prio_scores)

df["categories"] = all_labels
df["scores"] = all_scores
df["main_category"] = all_prio_scores

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDone. Results saved to: {OUTPUT_PATH}")