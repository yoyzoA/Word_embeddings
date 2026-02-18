import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


# ----------------------------
# Config (edit these paths)
# ----------------------------
JSONL_PATH = "ptb_pairs.jsonl"      # your row-wise JSONL
BASE_MODEL_NAME = "intfloat/e5-base-v2"
LORA_ADAPTER_PATH = "out/e5_lora"   # folder where adapter_model.safetensors exists

K_FARTHEST_BASES = 10
MAX_LEN = 128
BATCH_SIZE = 64

# For speed: t-SNE can be slow; this is okay for ~few hundred points
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42


# ----------------------------
# Data loading
# ----------------------------
def load_rows(jsonl_path):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(obj)
    return rows


def build_clusters(rows):
    """
    Returns:
      base_by_id: {id: base_phrase}
      variants_by_id: {id: [variant1, ...]}
    """
    base_by_id = {}
    variants_by_id = defaultdict(list)

    for r in rows:
        i = int(r["base_phrase_id"])
        base = r["base_phrase"]
        var = r["variant"]

        # base phrase repeats in file; keep first
        if i not in base_by_id:
            base_by_id[i] = base
        variants_by_id[i].append(var)

    # keep only clusters with variants
    keep_ids = [i for i in base_by_id if len(variants_by_id[i]) > 0]
    base_by_id = {i: base_by_id[i] for i in keep_ids}
    variants_by_id = {i: variants_by_id[i] for i in keep_ids}
    return base_by_id, variants_by_id


# ----------------------------
# Embedding helpers (E5)
# ----------------------------
def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)


@torch.no_grad()
def embed_texts(model, tokenizer, texts, device, max_len=128, batch_size=64):
    embs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        toks = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_len, return_tensors="pt"
        )
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(**toks, return_dict=True)
        emb = mean_pool(out.last_hidden_state, toks["attention_mask"])
        emb = F.normalize(emb, dim=-1)
        embs.append(emb.cpu().numpy())
    return np.vstack(embs)


def cosine_distance_matrix(X):
    # X is L2-normalized, so cosine similarity is dot product
    sim = X @ X.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def greedy_farthest_selection(dist, k):
    """
    dist: [N,N] distance matrix
    Returns indices of k items that are mutually far apart (greedy max-min).
    """
    N = dist.shape[0]
    # start from the farthest pair
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [i, j]

    while len(selected) < k:
        # choose point maximizing its minimum distance to selected set
        min_d = dist[:, selected].min(axis=1)
        for s in selected:
            min_d[s] = -1  # don't re-pick selected
        nxt = int(np.argmax(min_d))
        selected.append(nxt)

    return selected[:k]


# ----------------------------
# Visualization
# ----------------------------

def compute_cluster_stats(model, tokenizer, device, base_by_id, variants_by_id, chosen_ids, tag):
    texts = []
    labels = []

    for cid in chosen_ids:
        texts.append(f"query: {base_by_id[cid]}")
        labels.append(cid)
        for v in variants_by_id[cid]:
            texts.append(f"query: {v}")
            labels.append(cid)

    X = embed_texts(model, tokenizer, texts, device)

    intra = []
    inter = []

    for i in range(len(X)):
        for j in range(i+1, len(X)):
            cos = np.dot(X[i], X[j])
            if labels[i] == labels[j]:
                intra.append(cos)
            else:
                inter.append(cos)

    print(f"\n===== {tag} =====")
    print("Mean intra-cluster cosine:", np.mean(intra))
    print("Mean inter-cluster cosine:", np.mean(inter))
    print("Std intra:", np.std(intra))
    print("Std inter:", np.std(inter))


def plot_2d(points_2d, labels, title, out_path):
    """
    points_2d: [M,2]
    labels: list of cluster labels (base id) length M
    """
    unique = list(dict.fromkeys(labels))  # preserve order
    label_to_idx = {lab: idx for idx, lab in enumerate(unique)}

    plt.figure()
    # matplotlib default color cycle will handle different clusters
    for lab in unique:
        idxs = [i for i, x in enumerate(labels) if x == lab]
        pts = points_2d[idxs]
        plt.scatter(pts[:, 0], pts[:, 1], label=str(lab), s=18)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(markerscale=1.2, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_one_model(model, tokenizer, device, base_by_id, variants_by_id, chosen_ids, tag):
    """
    For chosen base IDs:
      - embed base + variants
      - run PCA(2) and t-SNE(2)
      - save plots
    """
    # Build points
    texts = []
    labels = []
    point_type = []  # "base" or "variant" (optional)

    for i in chosen_ids:
        base = f"query: {base_by_id[i]}"
        texts.append(base)
        labels.append(i)
        point_type.append("base")

        for v in variants_by_id[i]:
            texts.append(f"query: {v}")
            labels.append(i)
            point_type.append("variant")

    X = embed_texts(model, tokenizer, texts, device, max_len=MAX_LEN, batch_size=BATCH_SIZE)

    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    plot_2d(
        X_pca, labels,
        title=f"{tag} — PCA (2D) — {len(chosen_ids)} clusters",
        out_path=f"viz_{tag}_pca.png"
    )

    # t-SNE to 2D (use PCA init for stability)
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=min(TSNE_PERPLEXITY, max(5, (len(X) - 1) // 3)),
        random_state=TSNE_RANDOM_STATE
    )
    X_tsne = tsne.fit_transform(X)
    plot_2d(
        X_tsne, labels,
        title=f"{tag} — t-SNE (2D) — {len(chosen_ids)} clusters",
        out_path=f"viz_{tag}_tsne.png"
    )

    print(f"[OK] Saved: viz_{tag}_pca.png and viz_{tag}_tsne.png")


def main():
    rows = load_rows(JSONL_PATH)
    base_by_id, variants_by_id = build_clusters(rows)

    ids = sorted(base_by_id.keys())
    base_texts = [f"query: {base_by_id[i]}" for i in ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- BEFORE (base E5) ----
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModel.from_pretrained(BASE_MODEL_NAME).to(device)
    base_model.eval()

    base_embs_before = embed_texts(base_model, tok, base_texts, device, max_len=MAX_LEN, batch_size=BATCH_SIZE)
    dist = cosine_distance_matrix(base_embs_before)

    chosen_idx = greedy_farthest_selection(dist, K_FARTHEST_BASES)
    chosen_ids = [ids[i] for i in chosen_idx]
    print("Chosen base_phrase_id (farthest set):", chosen_ids)

    compute_cluster_stats(base_model, tok, device, base_by_id, variants_by_id, chosen_ids, "BEFORE")
    run_one_model(base_model, tok, device, base_by_id, variants_by_id, chosen_ids, tag="before")

    # ---- AFTER (E5 + LoRA adapter) ----
    base_model2 = AutoModel.from_pretrained(BASE_MODEL_NAME).to(device)
    after_model = PeftModel.from_pretrained(base_model2, LORA_ADAPTER_PATH).to(device)
    after_model.eval()
    
    compute_cluster_stats(after_model, tok, device, base_by_id, variants_by_id, chosen_ids, "AFTER")
    run_one_model(after_model, tok, device, base_by_id, variants_by_id, chosen_ids, tag="after")


if __name__ == "__main__":
    main()

# ===== BEFORE =====
# Mean intra-cluster cosine: 0.9650811
# Mean inter-cluster cosine: 0.6352734
# Std intra: 0.029151607
# Std inter: 0.023266328

# ===== AFTER =====
# Mean intra-cluster cosine: 1.0
# Mean inter-cluster cosine: 0.9999992
# Std intra: 7.14304e-08
# Std inter: 8.203284e-07