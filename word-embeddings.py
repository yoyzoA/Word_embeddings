import os
import random
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from nltk.corpus import wordnet
from nltk import download
from dotenv import load_dotenv
from google import genai
import pandas as pd
import matplotlib.pyplot as plt

download("wordnet")
download("omw-1.4")

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)


def generate_variants_multiword(sentence, n_replace=2, variants_per_phrase=10):
    """
    Generates phrases where n words are replaced at the same time.
    Returns a list of tuples (tag, tag, new_phrase) to be compatible
    with the interactive_approval logic.
    """
    words = sentence.split()
    all_variants = []

    synonym_map = {}
    for i, word in enumerate(words):
        synsets = wordnet.synsets(word)
        if not synsets:
            continue

        lemmas = set()
        for s in synsets:
            for lemma in s.lemmas():
                lemmas.add(lemma.name().replace("_", " "))

        synonyms = list(lemmas - {word})
        if synonyms:
            synonym_map[i] = synonyms  

    replaceable_positions = list(synonym_map.keys())

    if len(replaceable_positions) < n_replace:
        print(f"Not enough words with synonyms to replace {n_replace} at once.")
        return []

    for _ in range(variants_per_phrase):
        chosen_positions = random.sample(replaceable_positions, n_replace)

        new_words = words.copy()
        for pos in chosen_positions:
            syns = synonym_map[pos]
            if syns:
                new_words[pos] = random.choice(syns)

        new_phrase = " ".join(new_words)
        all_variants.append(("MULTI", "MULTI", new_phrase))

    return all_variants


def interactive_approval(sentence, n_replace=2, variants_per_phrase=10):
    print(f"\n🧩 Base phrase:\n{sentence}\n")

    variants = generate_variants_multiword(
        sentence,
        n_replace=n_replace,
        variants_per_phrase=variants_per_phrase
    )

    approved = [sentence]  

    print(f"🔎 Generated {len(variants)} candidate multi-word variants.\n")

    for (_tag1, _tag2, phrase) in variants:
        print(f"Variant:\n{phrase}")
        choice = input("Approve? (y/n/q): ").strip().lower()

        if choice == "y":
            approved.append(phrase)
            print("  ✓ Approved\n")
        elif choice == "n":
            print("  ✗ Rejected\n")
        elif choice == "q":
            print("  Stopping early for this base phrase.\n")
            break
        else:
            print("  Invalid input → skipping\n")

    return approved


def get_embeddings(phrases, model="text-embedding-004"):
    result = client.models.embed_content(model=model, contents=phrases)
    vectors = [emb.values for emb in result.embeddings]
    return np.array(vectors)


def compute_distances(phrases, embeddings):
    base = embeddings[0]
    distances = [0.0]  
    for emb in embeddings[1:]:
        distances.append(np.linalg.norm(emb - base))
    return distances


def visualize(all_groups, all_embeddings, all_distances):
    flattened_phrases = []
    flattened_clusters = []
    flattened_distances = []
    flattened_embed = []

    for i, group in enumerate(all_groups):
        cluster_name = f"Cluster {i+1}"
        for j, phrase in enumerate(group):
            flattened_phrases.append(phrase)
            flattened_clusters.append(cluster_name)
            flattened_distances.append(all_distances[i][j])
            flattened_embed.append(all_embeddings[i][j])

    flattened_embed = np.array(flattened_embed)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(flattened_embed)

    fig = px.scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        color=flattened_clusters,
        hover_name=flattened_phrases,
        hover_data={"Distance from Base": flattened_distances},
        title="Semantic Drift with Multi-word Synonym Replacement (Google GenAI embeddings)",
        width=1000,
        height=750
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="black")))
    fig.update_layout(hovermode="closest")
    fig.show()

def compute_cluster_statistics(all_groups, all_distances):
    print("\n======================================")
    print("SEMANTIC DRIFT STATISTICS")
    print("======================================\n")

    cluster_averages = []

    for i, (group, distances) in enumerate(zip(all_groups, all_distances), start=1):
        drift_values = distances[1:]  

        if len(drift_values) == 0:
            print(f"Cluster {i}: (no approved variants)")
            cluster_averages.append(0)
            continue

        avg = np.mean(drift_values)
        cluster_averages.append(avg)

        print(f"--- Cluster {i} ---")
        print(f"Base phrase: {group[0]}")
        print(f"Variants approved: {len(drift_values)}")
        print(f"Average drift:       {avg:.4f}")
        print(f"Minimum drift:       {np.min(drift_values):.4f}")
        print(f"Maximum drift:       {np.max(drift_values):.4f}")
        print(f"Std deviation:       {np.std(drift_values):.4f}\n")


    global_values = [d for cluster in all_distances for d in cluster[1:]]
    if len(global_values) > 0:
        global_avg = np.mean(global_values)
        print("======================================")
        print("OVERALL SEMANTIC DRIFT")
        print("======================================")
        print(f"Total variants approved: {len(global_values)}")
        print(f"Overall average drift:   {global_avg:.4f}")
        print(f"Overall min drift:       {np.min(global_values):.4f}")
        print(f"Overall max drift:       {np.max(global_values):.4f}")
        print(f"Overall std deviation:   {np.std(global_values):.4f}")
    else:
        print("No variants approved → no global stats.\n")

    return cluster_averages

def compute_cluster_statistics(all_groups, all_distances):
    cluster_stats = {}

    print("\n========== SEMANTIC DRIFT STATISTICS ==========\n")

    global_drift = []

    for i, (group, dist) in enumerate(zip(all_groups, all_distances), start=1):
        drift_vals = dist[1:]

        if not drift_vals:
            print(f"Cluster {i} — No valid variants\n")
            cluster_stats[i] = {
                "base": group[0],
                "avg": 0,
                "min": 0,
                "max": 0,
                "std": 0,
                "count": 0
            }
            continue

        avg = np.mean(drift_vals)
        mn = np.min(drift_vals)
        mx = np.max(drift_vals)
        sd = np.std(drift_vals)

        cluster_stats[i] = {
            "base": group[0],
            "avg": avg,
            "min": mn,
            "max": mx,
            "std": sd,
            "count": len(drift_vals)
        }

        print(f"--- Cluster {i} ---")
        print(f"Base phrase: {group[0]}")
        print(f"Variants: {len(drift_vals)}")
        print(f"Avg drift: {avg:.4f}")
        print(f"Min drift: {mn:.4f}")
        print(f"Max drift: {mx:.4f}")
        print(f"Std dev:   {sd:.4f}\n")

        global_drift.extend(drift_vals)

    if global_drift:
        global_stats = {
            "Overall Avg Drift": np.mean(global_drift),
            "Overall Min Drift": np.min(global_drift),
            "Overall Max Drift": np.max(global_drift),
            "Overall StdDev": np.std(global_drift),
            "Total Variant Count": len(global_drift)
        }

        print("==== OVERALL ====")
        for k, v in global_stats.items():
            print(f"{k}: {v:.4f}" if "Drift" in k else f"{k}: {v}")

    else:
        global_stats = {
            "Overall Avg Drift": 0,
            "Overall Min Drift": 0,
            "Overall Max Drift": 0,
            "Overall StdDev": 0,
            "Total Variant Count": 0
        }

    return cluster_stats, global_stats

def save_results_csv_and_plots(all_groups, all_distances, cluster_stats, global_stats):
    rows = []

    for i, (group, distances) in enumerate(zip(all_groups, all_distances), start=1):
        base_phrase = group[0]
        for phrase, dist in zip(group, distances):
            rows.append({
                "Cluster": f"Cluster {i}",
                "Base Phrase": base_phrase,
                "Variant Phrase": phrase,
                "Distance From Base": dist
            })

    df = pd.DataFrame(rows)

    cluster_rows = []
    for i, stats in cluster_stats.items():
        cluster_rows.append({
            "Cluster": f"Cluster {i}",
            "Base Phrase": stats["base"],
            "Average Drift": stats["avg"],
            "Min Drift": stats["min"],
            "Max Drift": stats["max"],
            "StdDev": stats["std"],
            "Count": stats["count"]
        })

    df_cluster = pd.DataFrame(cluster_rows)

    df_global = pd.DataFrame([global_stats])

    with pd.ExcelWriter("semantic_drift_results.xlsx") as writer:
        df.to_excel(writer, sheet_name="Phrase Variants", index=False)
        df_cluster.to_excel(writer, sheet_name="Cluster Stats", index=False)
        df_global.to_excel(writer, sheet_name="Overall Stats", index=False)

    print("\n Saved: semantic_drift_results.xlsx")

    all_drift_values = df["Distance From Base"].values
    plt.figure(figsize=(10, 6))
    plt.hist(all_drift_values, bins=20, color="skyblue", edgecolor="black")
    plt.title("Histogram of Semantic Drift Values")
    plt.xlabel("Drift Value (L2 norm)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("semantic_drift_histogram.png")
    plt.close()
    print("Saved: semantic_drift_histogram.png")

    plt.figure(figsize=(10, 6))
    plt.bar(df_cluster["Cluster"], df_cluster["Average Drift"], color="orange")
    plt.title("Average Semantic Drift per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Average Drift")
    plt.grid(axis="y")
    plt.savefig("cluster_average_drift.png")
    plt.close()
    print("Saved: cluster_average_drift.png")

if __name__ == "__main__":
    n_phrases = int(input("How many base phrases (N) do you want? N = "))
    n_replace = int(input("How many words to replace at the same time in each variant? n_replace = "))
    variants_per_phrase = int(input("How many variants per base phrase? variants_per_phrase = "))

    base_phrases = [input(f"Base phrase {i+1}: ") for i in range(n_phrases)]

    all_groups = []
    all_embeddings = []
    all_distances = []

    for i, phrase in enumerate(base_phrases, 1):
        print("\n======================================")
        print(f"PROCESSING BASE PHRASE {i}")
        print("======================================")

        group = interactive_approval(
            phrase,
            n_replace=n_replace,
            variants_per_phrase=variants_per_phrase
        )
        all_groups.append(group)

        embeddings = get_embeddings(group)
        all_embeddings.append(embeddings)

        distances = compute_distances(group, embeddings)
        all_distances.append(distances)

        print("\n📏 Distances from base phrase:")
        for ph, dist in zip(group, distances):
            print(f"{dist:.4f} → {ph}")

    visualize(all_groups, all_embeddings, all_distances)
    cluster_averages = compute_cluster_statistics(all_groups, all_distances)
    cluster_stats, global_stats = compute_cluster_statistics(all_groups, all_distances)

    save_results_csv_and_plots(
        all_groups,
        all_distances,
        cluster_stats,
        global_stats
    )


