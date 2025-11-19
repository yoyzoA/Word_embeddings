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
import seaborn as sns


# ================================
# INITIAL SETUP
# ================================
download("wordnet")
download("omw-1.4")

# Load E5 embedding model
print("Loading E5 model...")
tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")
model = AutoModel.from_pretrained("intfloat/e5-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ================================
# EMBEDDING FUNCTION (E5)
# ================================
def embed_sentence(sentence):
    """Embed a sentence using the E5 model and return a vector."""
    text = "query: " + sentence  # E5 expects this prefix
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0]  # CLS token embedding

    return emb[0].cpu().numpy()


def embed_batch(sentences):
    """Embed many sentences at once."""
    return np.vstack([embed_sentence(s) for s in sentences])


# ================================
# Multi-synonym generator
# ================================
def generate_variants_multi_n(sentence, n_replace_list, variants_per_n, seen_phrases):
    """
    Generates unique variants for multiple n_replace values.
    'seen_phrases' is a set of phrases already approved or rejected.
    """
    words = sentence.split()
    
    # Build synonym map
    synonym_map = {}
    for i, word in enumerate(words):
        synsets = wordnet.synsets(word)
        lemmas = set()
        for s in synsets:
            for lemma in s.lemmas():
                lemmas.add(lemma.name().replace("_", " "))
        synonyms = list(lemmas - {word})
        if synonyms:
            synonym_map[i] = synonyms

    replaceable_positions = list(synonym_map.keys())
    all_variants = set()  # ensure local uniqueness

    for n_replace in n_replace_list:
        if len(replaceable_positions) < n_replace:
            continue

        tries = 0
        max_tries = variants_per_n * 10  # prevents infinite loops

        while len([v for v in all_variants if v[1] == n_replace]) < variants_per_n and tries < max_tries:
            tries += 1

            new_words = words.copy()
            chosen_positions = random.sample(replaceable_positions, n_replace)

            for pos in chosen_positions:
                new_words[pos] = random.choice(synonym_map[pos])

            new_phrase = " ".join(new_words).strip()

            # Avoid duplicates across entire project (global uniqueness)
            if new_phrase.lower() in seen_phrases:
                continue

            # Also avoid duplicates in this local round
            all_variants.add((new_phrase, n_replace))

            seen_phrases.add(new_phrase.lower())

    return list(all_variants)



# ================================
# Manual approval of variants
# ================================
def manual_approval(base_phrase, n_replace_list, variants_per_n, seen_phrases):
    print(f"\nBase phrase:\n{base_phrase}")
    
    # Mark base phrase as seen
    seen_phrases.add(base_phrase.lower())

    variants = generate_variants_multi_n(
        base_phrase, n_replace_list, variants_per_n, seen_phrases
    )

    approved = [(base_phrase, 0)]

    for phrase, n_replaced in variants:
        print(f"\nVariant ({n_replaced} synonyms changed):\n{phrase}")
        choice = input("Approve? (y/n/q): ").strip().lower()

        if choice == "y":
            approved.append((phrase, n_replaced))
            print("  ✓ Approved")
        elif choice == "n":
            print("  ✗ Rejected")
            seen_phrases.add(phrase.lower())  # track rejection
        elif choice == "q":
            print("Stopping early for this base phrase.")
            break

    return approved



# ================================
# Distance + embedding processing
# ================================
def embed_and_distance(group):
    phrases = [p for p, n in group]
    n_replaced_list = [n for p, n in group]

    embeddings = embed_batch(phrases)
    base_vec = embeddings[0]

    distances = [0.0]  # base distance
    for e in embeddings[1:]:
        distances.append(np.linalg.norm(e - base_vec))

    return phrases, embeddings, distances, n_replaced_list


# ================================
# PCA Visualization
# ================================
def visualize(all_phrases, all_embeddings, all_distances, all_n_replaced, all_clusters):
    vecs = np.vstack(all_embeddings)
    reduced = PCA(n_components=2).fit_transform(vecs)

    df_plot = {
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "Phrase": all_phrases,
        "Cluster": all_clusters,
        "Synonyms_Replaced": all_n_replaced,
        "Distance": all_distances
    }

    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="Synonyms_Replaced",
        hover_name="Phrase",
        hover_data=["Cluster", "Distance"],
        title="Semantic Drift (E5 embeddings) Colored by Synonyms Replaced",
        width=1100,
        height=750
    )
    fig.show()


# ================================
# Cluster statistics
# ================================
def compute_statistics(all_groups, all_distances):
    cluster_stats = {}
    global_vals = []

    print("\n====== SEMANTIC DRIFT STATISTICS ======\n")

    for i, (group, dist) in enumerate(zip(all_groups, all_distances), start=1):
        dvals = dist[1:]  # exclude base phrase

        if len(dvals) == 0:
            cluster_stats[i] = {"base": group[0], "avg": 0, "min": 0, "max": 0, "std": 0, "count": 0}
            continue

        avg_d = np.mean(dvals)
        mn_d = np.min(dvals)
        mx_d = np.max(dvals)
        sd_d = np.std(dvals)

        cluster_stats[i] = {
            "base": group[0],
            "avg": avg_d,
            "min": mn_d,
            "max": mx_d,
            "std": sd_d,
            "count": len(dvals)
        }

        print(f"Cluster {i}:")
        print(f"Base phrase: {group[0]}")
        print(f"Variants: {len(dvals)}")
        print(f"Avg drift: {avg_d:.4f}, Min: {mn_d:.4f}, Max: {mx_d:.4f}, Std: {sd_d:.4f}\n")

        global_vals.extend(dvals)

    global_stats = {
        "Overall Avg Drift": np.mean(global_vals) if global_vals else 0,
        "Overall Min Drift": np.min(global_vals) if global_vals else 0,
        "Overall Max Drift": np.max(global_vals) if global_vals else 0,
        "Overall StdDev": np.std(global_vals) if global_vals else 0,
        "Total Variants": len(global_vals)
    }

    return cluster_stats, global_stats

def histogram_by_synonym_count(all_distances, all_n_replaced):
    """
    Plots a histogram where each 'n_replaced' value is shown as a separate color-coded group.
    """
    

    df = pd.DataFrame({
        "Distance": all_distances,
        "Synonyms_Replaced": all_n_replaced
    })

    # Remove base phrase rows (distance = 0, n_replaced = 0)
    df = df[df["Synonyms_Replaced"] > 0]

    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=df,
        x="Distance",
        hue="Synonyms_Replaced",
        bins=30,
        palette="viridis",
        kde=False,
        multiple="stack"  # or try "dodge" or "layer"
    )

    plt.title("Histogram of Semantic Drift by Number of Synonyms Replaced", fontsize=16)
    plt.xlabel("Distance from Base Embedding (L2 or Cosine)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(title="Synonyms Replaced")
    plt.tight_layout()
    plt.show()

# ================================
# Save to CSV & export plots
# ================================
def save_outputs(all_base_ids, all_phrases, all_distances, all_n_replaced, all_clusters,
                 cluster_stats, global_stats, base_phrases):

    df = pd.DataFrame({
        "Base_ID": all_base_ids,
        "Variant_Phrase": all_phrases,
        "Synonyms_Replaced": all_n_replaced,
        "Distance_From_Base": all_distances,
        "Cluster": all_clusters
    })

    df.to_csv("semantic_drift_phrases.csv", index=False)

    pd.DataFrame({
        "Base_ID": list(range(1, len(base_phrases)+1)),
        "Base_Phrase": base_phrases
    }).to_csv("semantic_drift_base_phrases.csv", index=False)

    pd.DataFrame.from_dict(cluster_stats, orient="index").to_csv("semantic_drift_clusters.csv")
    pd.DataFrame([global_stats]).to_csv("semantic_drift_overall.csv", index=False)

    print("\nCSV files saved.")

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df["Distance_From_Base"], bins=20, color="skyblue", edgecolor="black")
    plt.title("Histogram of Semantic Drift")
    plt.savefig("semantic_drift_histogram.png")
    plt.close()

    # Cluster barplot
    cluster_df = pd.DataFrame.from_dict(cluster_stats, orient="index")
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_df.index, cluster_df["avg"], color="orange")
    plt.title("Average Drift per Cluster")
    plt.savefig("cluster_average_drift.png")
    plt.close()


if __name__ == "__main__":
    print("=== SEMANTIC DRIFT EXPERIMENT (MANUAL + E5 MODEL) ===\n")

    # How many base phrases you want
    n_phrases = int(input("How many base phrases? "))

    # Enter each base phrase manually
    base_phrases = [input(f"Base phrase {i+1}: ") for i in range(n_phrases)]

    # Which levels of synonym replacement to try
    n_replace_list = list(map(int, input("Replace how many words at once? (e.g. 1 2 3): ").split()))

    # How many variants PER replacement level (per n_replace)
    variants_per_n = int(input("Variants per replacement level: "))

    print("\nStarting...\n")

    # =============================
    # GLOBAL STORAGE
    # =============================
    seen_phrases = set()      # Prevents duplicates across the entire experiment

    all_base_ids = []         # ID of which base phrase a variant belongs to
    all_phrases = []          # All phrases (base + variants)
    all_embeddings = []       # Embeddings
    all_distances = []        # L2 or cosine distances
    all_n_replaced = []       # Number of synonyms replaced for each phrase
    all_clusters = []         # Cluster name e.g. "Cluster 1"
    all_groups = []           # List of groups (base + approved variants)

    # =============================
    # PROCESS EACH BASE PHRASE
    # =============================
    for base_id, base in zip(range(1, n_phrases + 1), base_phrases):

        print("\n===============================")
        print(f"PROCESSING BASE PHRASE {base_id}")
        print("===============================\n")

        # APPROVAL STEP (now ensuring unique phrases)
        approved_group = manual_approval(
            base_phrase=base,
            n_replace_list=n_replace_list,
            variants_per_n=variants_per_n,
            seen_phrases=seen_phrases
        )

        # EMBED + DISTANCE
        phrases, emb, dist, n_rep = embed_and_distance(approved_group)

        # STORE RESULTS
        all_groups.append(approved_group)
        all_base_ids.extend([base_id] * len(phrases))
        all_phrases.extend(phrases)
        all_embeddings.extend(emb)
        all_distances.extend(dist)
        all_n_replaced.extend(n_rep)
        all_clusters.extend([f"Cluster {base_id}"] * len(phrases))

    # =============================
    # VISUALIZATION
    # =============================
    visualize(
        all_phrases, all_embeddings, all_distances,
        all_n_replaced, all_clusters
    )

    # =============================
    # STATISTICS
    # =============================
    cluster_stats, global_stats = compute_statistics(
        all_groups, all_distances
    )
    histogram_by_synonym_count(all_distances, all_n_replaced)

    # =============================
    # SAVE OUTPUTS
    # =============================
    save_outputs(
        all_base_ids, all_phrases, all_distances, all_n_replaced,
        all_clusters, cluster_stats, global_stats, base_phrases
    )

    print("\n=== DONE! ===\n")

