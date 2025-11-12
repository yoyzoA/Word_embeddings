

import os
import random
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from nltk import download
from dotenv import load_dotenv
from google import genai

# Setup
download("wordnet")
download("omw-1.4")

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

client = genai.Client(api_key=API_KEY)


# -----------------------------
# Generate variants
# -----------------------------
def generate_variants(sentence, max_synonyms_per_word=4):
    words = sentence.split()
    variants = []
    for i, word in enumerate(words):
        synsets = wordnet.synsets(word)
        if not synsets:
            continue
        lemmas = set(lemma.name().replace("_", " ") for s in synsets for lemma in s.lemmas())
        synonyms = list(lemmas - {word})
        random.shuffle(synonyms)
        for syn in synonyms[:max_synonyms_per_word]:
            new_sentence = " ".join([syn if j == i else w for j, w in enumerate(words)])
            variants.append((word, syn, new_sentence))
    return variants


# -----------------------------
# Interactive approval
# -----------------------------
def interactive_approval(sentence):
    print(f"\n🧩 Base phrase:\n{sentence}\n")
    variants = generate_variants(sentence)
    approved = [sentence]
    print(f"🔎 Generated {len(variants)} candidate variants.\n")

    for (orig, syn, phrase) in variants:
        print(f"Original word: '{orig}' → '{syn}'")
        print(f"Generated phrase:\n{phrase}")
        choice = input("Approve this variant? (y/n/q): ").strip().lower()
        if choice == "y":
            approved.append(phrase)
            print("✅ Approved.\n")
        elif choice == "n":
            print("⏭️ Skipped.\n")
        elif choice == "q":
            print("🛑 Stopping this phrase early.")
            break
        else:
            print("⚠️ Invalid input, skipping.\n")
    return approved


# -----------------------------
# Embeddings + visualization
# -----------------------------
def get_embeddings(phrases, model="text-embedding-004"):
    result = client.models.embed_content(model=model, contents=phrases)
    vectors = [emb.values for emb in result.embeddings]
    return np.array(vectors)

def visualize_embeddings(all_groups):
    all_phrases = [p for group in all_groups for p in group]
    group_labels = [f"Group {i+1}" for i, group in enumerate(all_groups) for _ in group]

    embeddings = get_embeddings(all_phrases)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    df = {
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "Phrase": all_phrases,
        "Group": group_labels
    }

    fig = px.scatter(
        df,
        x="x", y="y",
        color="Group",
        hover_data={"Phrase": True, "Group": True},
        title="Semantic Drift – Hover to See Phrase (Google GenAI Embeddings)",
        width=900, height=700
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(hovermode="closest")
    fig.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Enter three base phrases (press Enter after each):")
    base_phrases = [input(f"Phrase {i+1}: ") for i in range(3)]

    all_approved = []
    for i, phrase in enumerate(base_phrases, 1):
        print(f"\n===== PROCESSING PHRASE {i} =====")
        approved_group = interactive_approval(phrase)
        all_approved.append(approved_group)

    visualize_embeddings(all_approved)
