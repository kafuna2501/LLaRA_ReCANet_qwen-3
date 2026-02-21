import argparse
from pathlib import Path

import torch

from recommender.ReCANet import ReCANetEmbedder


def parse_args():
    parser = argparse.ArgumentParser(description="Build ReCANet .pt embedder from .weights.h5")
    parser.add_argument("--weights_path", required=True, type=str, help="Path to ReCANet .weights.h5")
    parser.add_argument("--data_dir", required=True, type=str, help="Path to ReCANet dataset dir (train/valid/test_baskets.csv)")
    parser.add_argument("--out_pt", default="rec_model/recanet_embedder.pt", type=str, help="Output .pt path")
    parser.add_argument("--recanet_dir", default="recanet-main", type=str, help="Path to recanet-main")

    parser.add_argument("--user_embed_size", default=32, type=int)
    parser.add_argument("--item_embed_size", default=128, type=int)
    parser.add_argument("--h1", default=64, type=int)
    parser.add_argument("--h2", default=64, type=int)
    parser.add_argument("--h3", default=64, type=int)
    parser.add_argument("--h4", default=64, type=int)
    parser.add_argument("--h5", default=64, type=int)
    parser.add_argument("--history_len", default=20, type=int)
    parser.add_argument("--basket_count_min", default=3, type=int)
    parser.add_argument("--min_item_count", default=5, type=int)
    parser.add_argument("--job_id", default=1, type=int)
    parser.add_argument("--padding_id", default=866, type=int)
    parser.add_argument("--use_next_only", action="store_true")
    parser.add_argument("--device", default="cpu", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights_path)
    data_dir = Path(args.data_dir)
    out_pt = Path(args.out_pt)

    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    embedder = ReCANetEmbedder(
        weights_path=str(weights_path),
        data_dir=str(data_dir),
        recanet_dir=str(args.recanet_dir),
        user_embed_size=args.user_embed_size,
        item_embed_size=args.item_embed_size,
        h1=args.h1,
        h2=args.h2,
        h3=args.h3,
        h4=args.h4,
        h5=args.h5,
        history_len=args.history_len,
        basket_count_min=args.basket_count_min,
        min_item_count=args.min_item_count,
        job_id=args.job_id,
        padding_id=args.padding_id,
        use_next_only=args.use_next_only,
        device=args.device,
    )

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedder, str(out_pt))
    print(f"saved: {out_pt}")


if __name__ == "__main__":
    main()
