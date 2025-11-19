import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from nltk.corpus import wordnet
from nltk import download
from transformers import AutoTokenizer, AutoModel
import torch

# ============================================
# SETUP
# ============================================
download("wordnet")
download("omw-1.4")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading E5-base embedding model...")
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")
model = AutoModel.from_pretrained("intfloat/e5-base")
model.to(device)


# ============================================
# E5 EMBEDDING FUNCTION
# ============================================
def embed_sentence(sentence):
    text = "query: " + sentence
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0]  # CLS token

    return emb[0].cpu().numpy()


def embed_batch(sent_list):
    return np.vstack([embed_sentence(s) for s in sent_list])


# ============================================
# LOAD ParaNMT-50M FILE
# ============================================
def load_paranmt_dataset(path, limit_per_base=10, max_bases=50):
    """
    Loads ParaNMT-50M variant with ONLY TWO columns:
    <reference>\t<paraphrase>
    """
    data = {}
    count = {}

    print("Loading ParaNMT...")

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split("\t")

            # Expect exactly 2 fields
            if len(parts) != 2:
                continue

            ref, para = parts
            score = 1.0  # synthetic placeholder score

            if ref not in data:
                data[ref] = []
                count[ref] = 0

            if count[ref] < limit_per_base:
                data[ref].append((para, score))
                count[ref] += 1

            if len(data) >= max_bases:
                break

    print(f"Loaded {len(data)} base sentences.")
    return data



# ============================================
# PREPARE GROUPS FOR BASE PHRASES
# ============================================
def prepare_paranmt_groups(paranmt_dict):
    base_phrases = []
    groups = []

    for base_id, (base, variants) in enumerate(paranmt_dict.items(), start=1):
        group = [(base, 1.0)]  # Base phrase: synthetic score = 1.0
        for para, score in variants:
            group.append((para, score))

        base_phrases.append(base)
        groups.append((base_id, group))

    return base_phrases, groups


# ============================================
# EMBEDDING + DISTANCE
# ============================================
def embed_and_distance_paranmt(group):
    phrases = [p for p, score in group]
    scores = [score for p, score in group]

    embeddings = embed_batch(phrases)
    base_vec = embeddings[0]

    distances = [0.0]
    for e in embeddings[1:]:
        distances.append(np.linalg.norm(e - base_vec))

    return phrases, embeddings, distances, scores


# ============================================
# VISUALIZATION
# ============================================
def visualize_paranmt(all_phrases, all_embeddings, all_distances, all_scores, all_clusters):
    vecs = np.vstack(all_embeddings)
    reduced = PCA(n_components=2).fit_transform(vecs)

    df = {
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "Phrase": all_phrases,
        "Cluster": all_clusters,
        "Embedding_Distance": all_distances,
        "Paragram_Score": all_scores
    }

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Cluster",  # <-- CLUSTER-BASED COLORING HERE
        hover_name="Phrase",
        hover_data=[
            "Cluster",
            "Embedding_Distance",
            "Paragram_Score"
        ],
        title="ParaNMT Semantic Drift (E5 Embeddings, Colored by Cluster)",
        width=1100,
        height=750
    )

    fig.show()



# ============================================
# SAVE CSV + HISTOGRAM + CLUSTER STATS
# ============================================
def save_outputs(base_ids, all_phrases, all_distances, all_scores, all_clusters, base_phrases):

    df = pd.DataFrame({
        "Base_ID": base_ids,
        "Phrase": all_phrases,
        "Paragram_Score": all_scores,
        "Embedding_Distance": all_distances,
        "Cluster": all_clusters
    })

    df.to_csv("paranmt_embeddings.csv", index=False)
    print("Saved: paranmt_embeddings.csv")

    pd.DataFrame({
        "Base_ID": range(1, len(base_phrases)+1),
        "Base_Phrase": base_phrases
    }).to_csv("paranmt_base_phrases.csv", index=False)
    print("Saved: paranmt_base_phrases.csv")

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df["Embedding_Distance"], bins=20, color="skyblue", edgecolor="black")
    plt.title("Histogram of Embedding Drift (E5)")
    plt.xlabel("L2 Distance from Base Phrase")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("embedding_drift_histogram.png")
    plt.close()
    print("Saved: embedding_drift_histogram.png")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":

    path = input("Enter path to ParaNMT dataset file: ")

    # LOAD PARA-NMT
    paranmt = load_paranmt_dataset(
        path,
        limit_per_base=10,   # paraphrases per reference sentence
        max_bases=40         # number of base sentences to visualize
    )

    base_phrases, groups = prepare_paranmt_groups(paranmt)

    # GLOBAL STORAGE
    all_base_ids = []
    all_phrases = []
    all_embeddings = []
    all_distances = []
    all_scores = []
    all_clusters = []

    # PROCESS EACH BASE SENTENCE
    for base_id, group in groups:

        print(f"Embedding Cluster {base_id}...")

        phrases, emb, dist, scores = embed_and_distance_paranmt(group)

        all_base_ids.extend([base_id] * len(phrases))
        all_phrases.extend(phrases)
        all_embeddings.extend(emb)
        all_distances.extend(dist)
        all_scores.extend(scores)
        all_clusters.extend([f"Cluster {base_id}"] * len(phrases))

    # VISUALIZE
    visualize_paranmt(
        all_phrases,
        all_embeddings,
        all_distances,
        all_scores,
        all_clusters
    )

    # SAVE DATA
    save_outputs(
        all_base_ids,
        all_phrases,
        all_distances,
        all_scores,
        all_clusters,
        base_phrases
    )

    print("\n=== DONE ===")
