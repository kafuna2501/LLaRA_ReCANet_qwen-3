import sys
from pathlib import Path

import os
import numpy as np
import pandas as pd
import torch


_DEFAULT_RECANET_DIR = Path(__file__).resolve().parent / "recanet-main"


class ReCANetEmbedder(torch.nn.Module):
    def __init__(
        self,
        weights_path,
        data_dir=None,
        pickle_dir=None,
        recanet_dir=None,
        user_embed_size=32,
        item_embed_size=128,
        h1=64,
        h2=64,
        h3=64,
        h4=64,
        h5=64,
        history_len=20,
        basket_count_min=3,
        min_item_count=5,
        job_id=1,
        padding_id=866,
        use_next_only=True,
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.padding_id = padding_id

        if "TF_USE_LEGACY_KERAS" not in os.environ:
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow is required to load ReCANet weights.") from exc

        recanet_dir = Path(recanet_dir) if recanet_dir else _DEFAULT_RECANET_DIR
        sys.path.insert(0, str(recanet_dir))
        sys.path.insert(0, str(recanet_dir / "models"))
        try:
            from models.mlp_v12 import MLPv12  # noqa: E402
        except Exception as exc:
            try:
                from mlp_v12 import MLPv12  # noqa: E402
            except Exception:
                raise ImportError(
                    f"Failed to import ReCANet models from {recanet_dir}."
                ) from exc

        if pickle_dir:
            train_baskets, valid_baskets, test_baskets = self._load_llara_pickles(
                pickle_dir, use_next_only=use_next_only
            )
            data_dir = str(pickle_dir)
        else:
            if data_dir is None:
                raise ValueError("Provide either data_dir (csv) or pickle_dir.")
            data_dir = str(data_dir)
            if not data_dir.endswith("/"):
                data_dir = data_dir + "/"
            train_baskets = pd.read_csv(data_dir + "train_baskets.csv")
            test_baskets = pd.read_csv(data_dir + "test_baskets.csv")
            valid_baskets = pd.read_csv(data_dir + "valid_baskets.csv")

        model = MLPv12(
            train_baskets,
            test_baskets,
            valid_baskets,
            data_dir,
            basket_count_min,
            min_item_count,
            user_embed_size,
            item_embed_size,
            h1,
            h2,
            h3,
            h4,
            h5,
            history_len,
            job_id=job_id,
        )

        model.model.load_weights(weights_path)

        embedding_layers = [
            layer for layer in model.model.layers
            if isinstance(layer, tf.keras.layers.Embedding)
        ]
        if len(embedding_layers) < 2:
            raise RuntimeError("Failed to locate item/user embedding layers.")

        # In ReCANet MLPv12, the first embedding is item and the second is user.
        item_weights = embedding_layers[0].get_weights()[0]
        user_weights = embedding_layers[1].get_weights()[0]

        self.item_id_mapper = model.item_id_mapper
        self.id_item_mapper = model.id_item_mapper
        self.user_id_mapper = model.user_id_mapper
        self.id_user_mapper = model.id_user_mapper

        item_emb = torch.tensor(item_weights, dtype=torch.float32)
        user_emb = torch.tensor(user_weights, dtype=torch.float32)
        self.item_embedding = torch.nn.Embedding.from_pretrained(item_emb, freeze=True)
        self.user_embedding = torch.nn.Embedding.from_pretrained(user_emb, freeze=True)
        self.item_embedding = self.item_embedding.to(self.device)
        self.user_embedding = self.user_embedding.to(self.device)
        self.item_embeddings = self.item_embedding
        self.user_embeddings = self.user_embedding
        self.register_buffer("mean_user_embedding", self._build_mean_user_embedding())

    def save_pt(self, path, weights_only=False):
        if weights_only:
            torch.save(self.item_embedding.weight.detach().cpu(), path)
        else:
            torch.save(self, path)

    def save(self, path):
        torch.save(self, path)

    def _map_ids(self, item_ids):
        if torch.is_tensor(item_ids):
            ids_np = item_ids.detach().cpu().numpy()
            out_device = item_ids.device
        else:
            ids_np = np.array(item_ids)
            out_device = self.device

        flat = ids_np.reshape(-1)
        mapped = np.array(
            [self.item_id_mapper.get(int(i), 0) for i in flat], dtype=np.int64
        ).reshape(ids_np.shape)
        return torch.from_numpy(mapped).to(out_device)

    def cacu_x(self, item_ids):
        mapped = self._map_ids(item_ids)
        return self.item_embedding(mapped)

    def _build_mean_user_embedding(self):
        known_user_indices = sorted(self.user_id_mapper.values())
        if len(known_user_indices) == 0:
            return torch.zeros(
                self.user_embedding.embedding_dim,
                dtype=self.user_embedding.weight.dtype,
                device=self.device
            )
        index_tensor = torch.tensor(known_user_indices, dtype=torch.long, device=self.device)
        return self.user_embedding(index_tensor).mean(dim=0)

    def _map_user_ids(self, user_ids):
        if torch.is_tensor(user_ids):
            ids_np = user_ids.detach().cpu().numpy()
            out_device = user_ids.device
        else:
            ids_np = np.array(user_ids)
            out_device = self.device

        flat = ids_np.reshape(-1)
        unknown_mask = np.array(
            [int(i) not in self.user_id_mapper for i in flat], dtype=bool
        ).reshape(ids_np.shape)
        mapped = np.array(
            [self.user_id_mapper.get(int(i), 0) for i in flat], dtype=np.int64
        ).reshape(ids_np.shape)
        mapped_tensor = torch.from_numpy(mapped).to(out_device)
        unknown_tensor = torch.from_numpy(unknown_mask).to(out_device)
        return mapped_tensor, unknown_tensor

    def cacu_u(self, user_ids):
        mapped, unknown_mask = self._map_user_ids(user_ids)
        user_embs = self.user_embedding(mapped)
        if unknown_mask.any():
            mask = unknown_mask
            while mask.ndim < user_embs.ndim:
                mask = mask.unsqueeze(-1)
            mean_emb = self.mean_user_embedding.to(user_embs.dtype)
            mean_shape = [1] * (user_embs.ndim - 1) + [-1]
            user_embs = torch.where(mask, mean_emb.view(*mean_shape), user_embs)
        return user_embs

    def _load_llara_pickles(self, pickle_dir, use_next_only=True):
        pickle_dir = str(pickle_dir)
        if not pickle_dir.endswith("/"):
            pickle_dir = pickle_dir + "/"

        train_path = Path(pickle_dir) / "train_data.df"
        test_path = Path(pickle_dir) / "Test_data.df"
        val_path = Path(pickle_dir) / "Cal_data.df"
        if not val_path.exists():
            val_path = Path(pickle_dir) / "Val_data.df"

        train_df = pd.read_pickle(train_path)
        test_df = pd.read_pickle(test_path)
        val_df = pd.read_pickle(val_path)

        train_baskets = self._df_to_baskets(train_df, split_offset=0, use_next_only=use_next_only)
        valid_baskets = self._df_to_baskets(val_df, split_offset=len(train_df) + 1, use_next_only=use_next_only)
        test_baskets = self._df_to_baskets(test_df, split_offset=len(train_df) + len(val_df) + 2, use_next_only=use_next_only)
        return train_baskets, valid_baskets, test_baskets

    def _df_to_baskets(self, df, split_offset=0, use_next_only=True):
        if use_next_only:
            rows = df[["member_id", "date", "next"]].copy()
            rows = rows.rename(columns={"member_id": "user_id", "next": "item_id"})
            rows["basket_id"] = np.arange(len(rows)) + split_offset
            return rows[["user_id", "date", "basket_id", "item_id"]]

        items_series = df["seq"].apply(
            lambda seq: [i for i in seq if i != self.padding_id]
        )
        rows = df[["member_id", "date"]].copy()
        rows["basket_id"] = np.arange(len(rows)) + split_offset
        rows = rows.loc[rows.index.repeat(items_series.str.len())].copy()
        rows["item_id"] = np.concatenate(items_series.values) if len(items_series) else []
        rows = rows.rename(columns={"member_id": "user_id"})
        return rows[["user_id", "date", "basket_id", "item_id"]]
