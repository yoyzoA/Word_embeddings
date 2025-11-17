import os
import random
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from google import genai
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")

client = genai.Client(api_key=API_KEY)

def generate_base_phrases(n_phrases):
    prompt = f"""
    Generate {n_phrases} short distinct English phrases, each describing a human action.
    Return them as a numbered list only.
    Example:
    1. The student reviews his notes.
    2. The athlete trains before sunrise.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    lines = response.text.strip().split("\n")
    phrases = [line.split(". ", 1)[1] for line in lines if ". " in line]
    return phrases


def generate_variants_gemini(base_phrase, n_replace, num_variants):
    prompt = f"""
    Given this phrase:
    "{base_phrase}"

    Generate {num_variants} rewritten variants where EXACTLY {n_replace}
    DIFFERENT words are replaced with meaningful synonyms.
    
    Preserve the general meaning, grammatical correctness, and person/tense.

    Return ONLY the rewritten sentences, one per line, no numbering.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    variants = [line.strip() for line in response.text.split("\n") if line.strip()]
    return variants


def auto_approve_variant(base, variant):
    prompt = f"""
    Base phrase:
    "{base}"

    Variant:
    "{variant}"

    Does the variant preserve the basic meaning of the base phrase
    while replacing words with appropriate synonyms?

    Answer strictly with "yes" or "no".
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    answer = response.text.strip().lower()
    return "yes" in answer


def embed_phrases(phrases):
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=phrases
    )
    return np.array([e.values for e in result.embeddings])


def compute_distances(embeddings):
    base = embeddings[0]
    return [0.0] + [np.linalg.norm(e - base) for e in embeddings[1:]]


def visualize(all_groups, all_embeddings, all_distances):
    phrases = []
    clusters = []
    distances = []
    vecs = []

    for i, (group, emb, dist) in enumerate(zip(all_groups, all_embeddings, all_distances)):
        cluster_name = f"Cluster {i+1}"
        for p, e, d in zip(group, emb, dist):
            phrases.append(p)
            clusters.append(cluster_name)
            distances.append(d)
            vecs.append(e)

    vecs = np.array(vecs)
    reduced = PCA(n_components=2).fit_transform(vecs)

    fig = px.scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        color=clusters,
        hover_name=phrases,
        hover_data={"Distance": distances},
        title="Semantic Drift – Fully Auto-Generated Multi-Synonym Variants",
        width=1100,
        height=750
    )

    fig.update_traces(marker=dict(size=10))
    fig.show()


def semantic_statistics(all_groups, all_distances):
    print("\n========== SEMANTIC DRIFT STATISTICS ==========\n")

    global_drift = []

    for i, (group, dist) in enumerate(zip(all_groups, all_distances), start=1):
        drift_vals = dist[1:]

        if not drift_vals:
            print(f"Cluster {i} — No valid variants\n")
            continue

        global_drift.extend(drift_vals)

        print(f"--- Cluster {i} ---")
        print(f"Base phrase: {group[0]}")
        print(f"Variants: {len(drift_vals)}")
        print(f"Avg drift: {np.mean(drift_vals):.4f}")
        print(f"Min drift: {np.min(drift_vals):.4f}")
        print(f"Max drift: {np.max(drift_vals):.4f}")
        print()

    if len(global_drift):
        print("==== OVERALL ====")
        print(f"Average drift: {np.mean(global_drift):.4f}")
        print(f"Min drift:     {np.min(global_drift):.4f}")
        print(f"Max drift:     {np.max(global_drift):.4f}")
        print(f"Std dev:       {np.std(global_drift):.4f}")

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
    
    N = 3                  
    N_REPLACE = 2           
    VARIANTS_PER_PHRASE = 10

    print("\n=== Generating Base Phrases ===")
    base_phrases = generate_base_phrases(N)
    for i, bp in enumerate(base_phrases, 1):
        print(f"{i}. {bp}")

    all_groups = []
    all_embeddings = []
    all_distances = []

    for base in base_phrases:
        print(f"\n=== Generating Variants for: {base} ===")

        raw_variants = generate_variants_gemini(base, N_REPLACE, VARIANTS_PER_PHRASE)

        approved = [base]
        for variant in raw_variants:
            if auto_approve_variant(base, variant):
                approved.append(variant)

        print(f"Approved {len(approved)-1} variants.")

        emb = embed_phrases(approved)
        dist = compute_distances(emb)

        all_groups.append(approved)
        all_embeddings.append(emb)
        all_distances.append(dist)

    visualize(all_groups, all_embeddings, all_distances)
    semantic_statistics(all_groups, all_distances)
    
    cluster_stats, global_stats = compute_cluster_statistics(all_groups, all_distances)

    save_results_csv_and_plots(
    all_groups,
    all_distances,
    cluster_stats,
    global_stats
    )

