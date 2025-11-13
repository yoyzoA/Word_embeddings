# filename: n_phrase_multiword_semantic_drift.py
# pip install google-genai python-dotenv nltk scikit-learn numpy plotly

import os
import random
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from nltk.corpus import wordnet
from nltk import download
from dotenv import load_dotenv
from google import genai

# -----------------------------
# Setup
# -----------------------------
download("wordnet")
download("omw-1.4")

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)


# -------------------------------------------------------
# 1) Multi-word variant generator
# -------------------------------------------------------
def generate_variants_multiword(sentence, n_replace=2, variants_per_phrase=10):
    """
    Generates phrases where n words are replaced at the same time.
    Returns a list of tuples (tag, tag, new_phrase) to be compatible
    with the interactive_approval logic.
    """
    words = sentence.split()
    all_variants = []

    # Precompute synonyms for each word
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
            synonym_map[i] = synonyms  # index -> list of synonyms

    replaceable_positions = list(synonym_map.keys())

    if len(replaceable_positions) < n_replace:
        print(f"⚠️ Not enough words with synonyms to replace {n_replace} at once.")
        return []

    for _ in range(variants_per_phrase):
        # pick n positions to replace
        chosen_positions = random.sample(replaceable_positions, n_replace)

        new_words = words.copy()
        for pos in chosen_positions:
            syns = synonym_map[pos]
            if syns:
                new_words[pos] = random.choice(syns)

        new_phrase = " ".join(new_words)
        all_variants.append(("MULTI", "MULTI", new_phrase))

    return all_variants


# -------------------------------------------------------
# 2) Human approval loop for ONE base phrase
# -------------------------------------------------------
def interactive_approval(sentence, n_replace=2, variants_per_phrase=10):
    print(f"\n🧩 Base phrase:\n{sentence}\n")

    variants = generate_variants_multiword(
        sentence,
        n_replace=n_replace,
        variants_per_phrase=variants_per_phrase
    )

    approved = [sentence]  # base always included

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


# -------------------------------------------------------
# 3) Embeddings from Google GenAI
# -------------------------------------------------------
def get_embeddings(phrases, model="text-embedding-004"):
    result = client.models.embed_content(model=model, contents=phrases)
    vectors = [emb.values for emb in result.embeddings]
    return np.array(vectors)


# -------------------------------------------------------
# 4) Compute L2 distances from base phrase
# -------------------------------------------------------
def compute_distances(phrases, embeddings):
    base = embeddings[0]
    distances = [0.0]  # base distance = 0
    for emb in embeddings[1:]:
        distances.append(np.linalg.norm(emb - base))
    return distances


# -------------------------------------------------------
# 5) Plot all clusters with Plotly
# -------------------------------------------------------
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


# -------------------------------------------------------
# 6) MAIN PROGRAM — N base phrases
# -------------------------------------------------------
if __name__ == "__main__":
    # Ask how many base phrases and how strong the drift should be
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
