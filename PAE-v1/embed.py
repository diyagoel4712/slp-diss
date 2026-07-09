"""
embed.py — Inference: produce accent embeddings from audio clips.

Usage:
    # Single file
    python embed.py --checkpoint checkpoints/best.pt --audio clip.wav

    # Directory → embeddings.json
    python embed.py --checkpoint checkpoints/best.pt --audio_dir data/test/

    # Visualise embedding space with UMAP
    python embed.py --checkpoint checkpoints/best.pt --audio_dir data/test/ --plot
"""

import json
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path

from model import AccentEncoder


def load_model(checkpoint_path: str, device: torch.device) -> AccentEncoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt["args"]

    # Load speaker map to get num_speakers
    spk_map_path = Path(checkpoint_path).parent / "speaker_map.json"
    with open(spk_map_path) as f:
        speaker_to_id = json.load(f)

    model = AccentEncoder(
        backbone=args["backbone"],
        embedding_dim=args["embedding_dim"],
        num_speakers=len(speaker_to_id),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def embed_audio(model: AccentEncoder, wav_path: str, device: torch.device) -> np.ndarray:
    """
    Embed a single .wav file.
    Returns: numpy array of shape (embedding_dim,)
    """
    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.mean(0).unsqueeze(0).to(device)  # (1, T)

    embedding = model.encode(waveform)  # (1, D)
    return embedding.squeeze(0).cpu().numpy()


@torch.no_grad()
def embed_directory(model: AccentEncoder, audio_dir: str, device: torch.device) -> dict:
    """Embed all .wav files in a directory. Returns {filename: embedding}."""
    results = {}
    for wav_path in sorted(Path(audio_dir).glob("*.wav")):
        emb = embed_audio(model, str(wav_path), device)
        results[wav_path.stem] = emb
        print(f"  {wav_path.stem}: {emb.shape}")
    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def visualise_embeddings(embeddings: dict, labels: dict = None):
    """
    UMAP projection of embeddings coloured by label (e.g. accent variety).
    labels: {utterance_id: "label_string"}
    """
    try:
        import umap
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("Install umap-learn and matplotlib for visualisation: pip install umap-learn matplotlib")
        return

    ids = list(embeddings.keys())
    vecs = np.stack([embeddings[i] for i in ids])

    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42, n_neighbors=15)
    coords = reducer.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels:
        unique_labels = sorted(set(labels.values()))
        cmap = cm.get_cmap("tab20", len(unique_labels))
        label_to_colour = {l: cmap(i) for i, l in enumerate(unique_labels)}

        for uid, (x, y) in zip(ids, coords):
            label = labels.get(uid, "unknown")
            colour = label_to_colour[label]
            ax.scatter(x, y, c=[colour], s=40, alpha=0.7)

        # Legend
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=label_to_colour[l], markersize=8, label=l)
            for l in unique_labels
        ]
        ax.legend(handles=handles, title="Accent", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=40, alpha=0.7)

    ax.set_title("Phonological Accent Embedding Space (UMAP projection)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig("embedding_space.png", dpi=150)
    print("Saved → embedding_space.png")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio",      help="Single .wav file")
    p.add_argument("--audio_dir",  help="Directory of .wav files")
    p.add_argument("--labels",     help="JSON file: {utterance_id: accent_label}")
    p.add_argument("--out",        default="embeddings.json")
    p.add_argument("--plot",       action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if args.audio:
        emb = embed_audio(model, args.audio, device)
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding (first 8 dims): {emb[:8]}")

    elif args.audio_dir:
        embeddings = embed_directory(model, args.audio_dir, device)
        # Save as JSON (lists for serializability)
        out = {k: v.tolist() for k, v in embeddings.items()}
        with open(args.out, "w") as f:
            json.dump(out, f)
        print(f"\nSaved {len(out)} embeddings → {args.out}")

        if args.plot:
            labels = None
            if args.labels:
                with open(args.labels) as f:
                    labels = json.load(f)
            visualise_embeddings(embeddings, labels)
    else:
        p.error("Provide --audio or --audio_dir")


if __name__ == "__main__":
    main()
