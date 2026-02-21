import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data

import pandas as pd
import random


class RetailData(data.Dataset):
    def __init__(
        self,
        data_dir=r"data/ref/retail",
        stage=None,
        sep=", ",
        no_augment=True,
    ):
        # safer than self.__dict__.update(locals()) which includes `self`
        args = locals()
        args.pop("self")
        self.__dict__.update(args)

        self.aug = (stage == "train") and not no_augment
        self.padding_item_id = 866
        self.check_files()

    def __len__(self):
        return len(self.session_data["seq"])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]

        # --- pretty history string (date-aware if available) ---
        history_dates = temp.get("history_dates", [])
        if history_dates:
            history_pairs = [
                f"{date}: {self.sep.join(seq_titles)}"
                for date, seq_titles in zip(history_dates, temp["history_seq_title"])
            ]
        else:
            history_pairs = [self.sep.join(seq) for seq in temp["history_seq_title"]]

        # --- flatten date-based history (past dates only) ---
        max_len = len(temp["seq"])  # keep same length as seq to avoid stack issues

        # temp["history_seq"] is list[list[int]] (past days only)
        hist_ids = [iid for day_seq in temp["history_seq"] for iid in day_seq]

        # fallback if empty (to avoid zero-length history)
        if len(hist_ids) == 0:
            hist_ids = temp["seq_unpad"][:]  # unpadded sequence

        # keep most recent max_len items
        hist_ids = hist_ids[-max_len:]
        len_hist = len(hist_ids)

        # left-pad to max_len
        hist_padded = [self.padding_item_id] * (max_len - len_hist) + hist_ids

        # names for prompt (no padding)
        hist_names = [self.item_id2name.get(iid, f"<UNK_{iid}>") for iid in hist_ids]
        raw_user_id = temp.get("member_id", 0)
        user_id = 0 if pd.isna(raw_user_id) else int(raw_user_id)

        sample = {
            "seq": temp["seq"],
            "seq_name": temp["seq_title"],
            "len_seq": temp["len_seq"],
            "seq_str": self.sep.join(temp["seq_title"]),

            "history_seq": temp["history_seq"],
            "history_seq_name": temp["history_seq_title"],
            "history_dates": history_dates,
            # now actually use history_pairs (date-aware)
            "history_str": " || ".join(history_pairs),

            "item_id": temp["next"],
            "item_name": temp["next_item_name"],
            "correct_answer": temp["next_item_name"],

            # safe access (date column may not exist in some datasets)
            "date": temp.get("date", None),
            "user_id": user_id,

            # flattened history for prompt+embedding alignment
            "hist_seq": hist_padded,
            "hist_seq_name": hist_names,
            "len_hist_seq": len_hist,
        }
        return sample

    def check_files(self):
        self.item_id2name = self.get_retail_id2name()

        if self.stage == "train":
            filename = "train_data.df"
        elif self.stage == "val":
            filename = "Val_data.df"
        elif self.stage == "test":
            filename = "Test_data.df"
        else:
            raise ValueError(f"stage must be one of ['train','val','test'], got: {self.stage}")

        data_path = op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)

    def get_retail_id2name(self):
        retail_id2name = dict()
        item_path = op.join(self.data_dir, "id2name.txt")
        with open(item_path, "r") as f:
            for l in f.readlines():
                ll = l.strip("\n").split(",")
                retail_id2name[int(ll[0])] = ll[1].strip()
        return retail_id2name

    def session_data4frame(self, datapath, retail_id2name):
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data["len_seq"] >= 10].copy()  # CHANGE HERE FOR len_req

        def remove_padding(xx):
            return [v for v in xx if v != self.padding_item_id]

        train_data["seq_unpad"] = train_data["seq"].apply(remove_padding)

        def seq_to_title(x):
            return [retail_id2name[x_i] for x_i in x]

        train_data["seq_title"] = train_data["seq_unpad"].apply(seq_to_title)

        def next_item_title(x):
            return retail_id2name[x]

        train_data["next_item_name"] = train_data["next"].apply(next_item_title)

        # --- Build history that only includes sequences prior to the current date ---
        if {"member_id", "date"}.issubset(train_data.columns):
            train_data = train_data.sort_values(["member_id", "date"])

            def build_member_history(group):
                history_seq_col = []
                history_seq_title_col = []
                history_dates_col = []

                # Track every sequence observed for each prior date.
                date_to_seqs = {}
                date_to_titles = {}
                ordered_dates = []

                for (_, row) in group.iterrows():
                    cur_date = row["date"]
                    seq = row["seq_unpad"]
                    seq_title = row["seq_title"]

                    # Build history from dates strictly earlier than cur_date.
                    # When multiple rows share the same date we randomly sample one
                    # sequence for that date so history never repeats a day more than once.
                    seq_history = []
                    title_history = []
                    date_history = []

                    for date_key in ordered_dates:
                        if date_key >= cur_date:
                            continue
                        candidates = date_to_seqs[date_key]
                        title_candidates = date_to_titles[date_key]
                        idx = random.randrange(len(candidates))
                        seq_history.append(candidates[idx][:])
                        title_history.append(title_candidates[idx][:])
                        date_history.append(date_key)

                    history_seq_col.append(seq_history)
                    history_seq_title_col.append(title_history)
                    history_dates_col.append(date_history)

                    if cur_date not in date_to_seqs:
                        date_to_seqs[cur_date] = []
                        date_to_titles[cur_date] = []
                        ordered_dates.append(cur_date)

                    date_to_seqs[cur_date].append(seq)
                    date_to_titles[cur_date].append(seq_title)

                return pd.DataFrame(
                    {
                        "history_seq": history_seq_col,
                        "history_seq_title": history_seq_title_col,
                        "history_dates": history_dates_col,
                    },
                    index=group.index,
                )

            history_df = (
                train_data.groupby("member_id", group_keys=False).apply(build_member_history)
            )
            train_data[["history_seq", "history_seq_title", "history_dates"]] = history_df
        else:
            # Keep backward compatibility when either column is missing.
            train_data["history_seq"] = train_data["seq_unpad"].apply(lambda seq: [seq])
            train_data["history_seq_title"] = train_data["seq_title"].apply(lambda seq: [seq])
            train_data["history_dates"] = [[] for _ in range(len(train_data))]

        return train_data
