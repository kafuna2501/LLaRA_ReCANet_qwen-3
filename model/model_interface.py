import inspect
import torch
import importlib
from pathlib import Path
import re
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl

from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from pandas.core.frame import DataFrame
import os.path as op
import os
from optims import LinearWarmupCosineLRScheduler
import numpy as np
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel

class MInterface(pl.LightningModule):
    def __init__(self,
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        self._warned_missing_attention_mask = False
        self._warned_missing_user_embedding = False
        self.load_llm(self.hparams.llm_path)
        self.load_rec_model(self.hparams.rec_model_path)
        self.load_projector()
        self.all_item_names = self.load_all_item_names(getattr(self.hparams, "data_dir", None))
        self.all_item_name_set = set(self.all_item_names)

    def forward(self, batch):
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        ) # [batch_size, max_len]
        token_type_ids = getattr(batch["tokens"], "token_type_ids", None)
        if token_type_ids is not None:
            targets = targets.masked_fill((token_type_ids == 0), -100)
        input_embeds = self.wrap_emb(batch)
        attention_mask = self._build_attention_mask(batch["tokens"])
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False
        )
        return outputs

    def generate(self, batch,temperature=0.8,do_sample=False,num_beams=1,max_gen_length=64,min_gen_length=1,repetition_penalty=1.0,length_penalty=1.0, num_return_sequences=1):
        input_embeds = self.wrap_emb(batch)
        attention_mask = self._build_attention_mask(batch["tokens"])
        generation_kwargs = dict(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences
            )
        # temperature is only meaningful when sampling is enabled.
        if do_sample:
            generation_kwargs["temperature"] = temperature
        generate_ids = self.llama_model.generate(**generation_kwargs)
        output_text=self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs=[text.strip() for text in output_text]
        return outputs

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        projectors = [self.projector]
        if self.user_projector is not None:
            projectors.append(self.user_projector)
        requires_grad = not batch["flag"]
        for projector in projectors:
            for _, param in projector.named_parameters():
                param.requires_grad = requires_grad
        out = self(batch)
        loss = self.configure_loss(out)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_content={
            "generate_raw":[],
            "generate_item":[],
            "explanation":[],
            "real_item":[],
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        generate_output = self.generate(batch)
        output=[]
        for i, generate_raw in enumerate(generate_output):
            real_item = batch['correct_answer'][i]
            generate_item, explanation = self._extract_item_and_explanation(generate_raw)
            output.append((str(generate_raw), generate_item, explanation, real_item))
        return output

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        for generate_raw, generate_item, explanation, real_item in outputs:
            self.val_content["generate_raw"].append(generate_raw)
            self.val_content["generate_item"].append(generate_item)
            self.val_content["explanation"].append(explanation)
            self.val_content["real_item"].append(real_item)

    def on_validation_epoch_end(self):
        df=DataFrame(self.val_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'valid.csv'))
        prediction_valid_ratio,hr=self.calculate_hr1(self.val_content)
        metric=hr*prediction_valid_ratio
        self.log('val_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_content={
            "generate_raw":[],
            "generate_item":[],
            "explanation":[],
            "real_item":[],
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        generate_output = self.generate(batch)
        output=[]
        for i, generate_raw in enumerate(generate_output):
            real_item = batch['correct_answer'][i]
            generate_item, explanation = self._extract_item_and_explanation(generate_raw)
            output.append((str(generate_raw), generate_item, explanation, real_item))
        return output

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        for generate_raw, generate_item, explanation, real_item in outputs:
            self.test_content["generate_raw"].append(generate_raw)
            self.test_content["generate_item"].append(generate_item)
            self.test_content["explanation"].append(explanation)
            self.test_content["real_item"].append(real_item)

    def on_test_epoch_end(self):
        df=DataFrame(self.test_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'test.csv'))

        #Debug start
        print("===== Sample test predictions (first 5) =====")
        n_show = min(100, len(self.test_content["generate_raw"]))
        for i in range(n_show):
            gen_raw = self.test_content["generate_raw"][i]
            gen_item = self.test_content["generate_item"][i]
            explanation = self.test_content["explanation"][i]
            real_item = self.test_content["real_item"][i]
            print(f"[{i}]")
            print(f"  raw  : {gen_raw}")
            print(f"  item : {gen_item}")
            print(f"  expl : {explanation}")
            print(f"  real : {real_item}")
            print("------------------------------------------")
        #Debug end

        prediction_valid_ratio,hr=self.calculate_hr1(self.test_content)
        metric=hr*prediction_valid_ratio
        self.log('test_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optim_groups = [
            {'params': self.projector.parameters(), 'lr': self.hparams.lr, 'weight_decay': weight_decay},
        ]
        if self.user_projector is not None:
            optim_groups.append(
                {'params': self.user_projector.parameters(), 'lr': self.hparams.lr, 'weight_decay': weight_decay}
            )
        optim_groups.append({'params': self.llama_model.parameters(), 'lr': self.hparams.lr})
        optimizer = torch.optim.Adam(optim_groups)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            max_step = self.trainer.max_steps
            warmup_steps = max_step // 20
            print(f'max_step: {max_step}')
            print(f'warmup_steps: {warmup_steps}')
            if self.hparams.lr_scheduler == 'cosine':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                                                  max_step=max_step,
                                                  min_lr=self.hparams.lr_decay_min_lr,
                                                  init_lr=self.hparams.lr,
                                                  warmup_steps=warmup_steps,
                                                  warmup_start_lr=self.hparams.lr_warmup_start_lr)
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer

    def configure_loss(self, out, labels=None):
        loss = self.hparams.loss.lower()
        if loss == 'lm':
            return out.loss
        else:
            raise ValueError("Invalid Loss Type!")

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == 'part':
            checkpoint.pop('optimizer_states')
            to_be_removed = []
            for key, value in checkpoint['state_dict'].items():
                try:
                    if not self.get_parameter(key).requires_grad:
                        to_be_removed.append(key)
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
        elif self.hparams.save == 'all':
            pass

    def load_llm(self, llm_path):
        print('Loading LLM')
        trust_remote_code = getattr(self.hparams, "trust_remote_code", False)
        llm_path, gguf_file = self._resolve_llm_path_and_gguf_file(llm_path)
        tokenizer_path = getattr(self.hparams, "tokenizer_path", None) or llm_path
        self.llama_tokenizer = self._load_tokenizer_with_fallback(
            tokenizer_path=tokenizer_path,
            model_path=llm_path,
            trust_remote_code=trust_remote_code,
        )
        # Ensure that padding uses a dedicated token instead of sharing EOS so we can
        # reliably rebuild attention masks even if the dataloader forgets to supply
        # them (which was causing the model to generate the same "Tags" string for
        # every test example).
        tokenizer_resized = False
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer_resized = True
        self.llama_tokenizer.pad_token = self.llama_tokenizer.pad_token or '[PAD]'
        self.llama_tokenizer.padding_side = "right"
        added_tokens = self.llama_tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[PH]','[HistoryEmb]','[ItemEmb]','[UserEmb]']}
        )
        tokenizer_resized = tokenizer_resized or (added_tokens > 0)
        llm_torch_dtype = self._resolve_torch_dtype()
        llm_load_kwargs = {
            "torch_dtype": llm_torch_dtype,
            "trust_remote_code": trust_remote_code,
        }
        if gguf_file:
            llm_load_kwargs["gguf_file"] = gguf_file
            print(f"Loading GGUF model: {gguf_file}")

        self.llama_model = AutoModelForCausalLM.from_pretrained(llm_path, **llm_load_kwargs)
        if tokenizer_resized:
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        self.llama_model.config.pad_token_id = self.llama_tokenizer.pad_token_id
        if self.hparams.llm_tuning == 'lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llama_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    lora_target_modules = self._resolve_lora_target_modules()
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=int(self.hparams.lora_r),
                                             lora_alpha=int(self.hparams.lora_alpha),
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=lora_target_modules)
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        elif self.hparams.llm_tuning == 'freeze':
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        elif self.hparams.llm_tuning == 'freeze_lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llama_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    lora_target_modules = self._resolve_lora_target_modules()
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=int(self.hparams.lora_r),
                                             lora_alpha=int(self.hparams.lora_alpha),
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=lora_target_modules)
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            self.llama_model.print_trainable_parameters()
        else:
            raise NotImplementedError()

        print('Loading LLM Done')

    def _load_tokenizer_with_fallback(self, tokenizer_path, model_path, trust_remote_code):
        candidates = [tokenizer_path]
        model_path_l = str(model_path).lower() if model_path is not None else ""
        tokenizer_path_l = str(tokenizer_path).lower() if tokenizer_path is not None else ""
        if "gpt-oss" in model_path_l or "gpt-oss" in tokenizer_path_l:
            candidates.append("openai/gpt-oss-20b")

        seen = set()
        unique_candidates = []
        for c in candidates:
            if c and c not in seen:
                unique_candidates.append(c)
                seen.add(c)

        errors = []
        for candidate in unique_candidates:
            for use_fast in (False, True):
                try:
                    print(f"Loading tokenizer from {candidate} (use_fast={use_fast})")
                    return AutoTokenizer.from_pretrained(
                        candidate,
                        use_fast=use_fast,
                        trust_remote_code=trust_remote_code
                    )
                except Exception as exc:
                    errors.append(f"{candidate} use_fast={use_fast}: {exc}")

        error_text = "\n".join(errors[:6])
        raise RuntimeError(
            "Failed to load tokenizer. Install `tiktoken` and/or set `--tokenizer_path` "
            "(for GPT-OSS, try `--tokenizer_path openai/gpt-oss-20b`).\n"
            f"Attempts:\n{error_text}"
        )

    def _resolve_llm_path_and_gguf_file(self, llm_path):
        gguf_file = getattr(self.hparams, "llm_gguf_file", None)
        if llm_path and llm_path.lower().endswith(".gguf") and gguf_file is None:
            llm_file_path = Path(llm_path)
            return str(llm_file_path.parent), llm_file_path.name
        return llm_path, gguf_file

    def _resolve_torch_dtype(self):
        dtype_name = str(getattr(self.hparams, "llm_dtype", "bf16")).lower()
        dtype_map = {
            "auto": "auto",
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported llm_dtype: {dtype_name}")
        return dtype_map[dtype_name]

    def _resolve_lora_target_modules(self):
        override = getattr(self.hparams, "lora_target_modules", None)
        if override:
            modules = [m.strip() for m in override.split(",") if m.strip()]
            if not modules:
                raise ValueError("lora_target_modules was provided but no valid module names were parsed.")
            print(f"LoRA target modules (manual): {modules}")
            return modules

        # Auto-detect from known linear submodule names to support multiple model families.
        preferred_names = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "wq", "wk", "wv", "wo",
            "query_key_value",
            "c_attn", "c_proj",
            "fc_in", "fc_out",
        ]
        linear_leaf_names = set()
        for name, module in self.llama_model.named_modules():
            if isinstance(module, nn.Linear):
                linear_leaf_names.add(name.split(".")[-1])

        modules = [name for name in preferred_names if name in linear_leaf_names]
        if not modules:
            available = sorted(linear_leaf_names)
            preview = available[:30]
            raise ValueError(
                "Failed to auto-detect LoRA target modules. "
                f"Available linear module names (first 30): {preview}. "
                "Set --lora_target_modules explicitly."
            )
        print(f"LoRA target modules (auto): {modules}")
        return modules

    def load_projector(self):
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')

        item_rec_size = self._infer_item_rec_size()
        if item_rec_size is None:
            item_rec_size = self.hparams.rec_size
        self.projector = self.instancialize(
            Model,
            rec_size=item_rec_size,
            llm_size=self.llama_model.config.hidden_size
        )

        user_rec_size = self._infer_user_rec_size()
        if user_rec_size is not None:
            self.user_projector = self.instancialize(
                Model,
                rec_size=user_rec_size,
                llm_size=self.llama_model.config.hidden_size
            )
        else:
            self.user_projector = None

    def instancialize(self, Model, **other_args):
        sig = inspect.signature(Model.__init__)
        class_args = [p for p in sig.parameters.keys()][1:]  # self を除夁E


        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    def load_rec_model(self, rec_model_path):
        print('Loading Rec Model')
        self.rec_model = torch.load(rec_model_path, map_location=self.device, weights_only=False)
        self.rec_model = self.rec_model.to(self.device)
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False
        print('Loding Rec model Done')

    def _normalize_item_name(self, text):
        if text is None:
            return ""
        return str(text).strip().lower()

    def load_all_item_names(self, data_dir):
        if not data_dir:
            raise ValueError("data_dir is required to load id2name.txt for valid-ratio evaluation.")
        id2name_path = op.join(data_dir, "id2name.txt")
        if not op.isfile(id2name_path):
            raise FileNotFoundError(f"id2name.txt not found: {id2name_path}")

        names = []
        with open(id2name_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(",", 1)
                if len(parts) < 2:
                    continue
                name = self._normalize_item_name(parts[1])
                if name:
                    names.append(name)

        # Keep file order while removing duplicates.
        unique_names = list(dict.fromkeys(names))
        if not unique_names:
            raise ValueError(f"No valid item names found in {id2name_path}")
        return unique_names

    def _embedding_dim(self, embedding_layer):
        if embedding_layer is None:
            return None
        if hasattr(embedding_layer, "embedding_dim"):
            return int(embedding_layer.embedding_dim)
        if hasattr(embedding_layer, "weight") and embedding_layer.weight is not None:
            if embedding_layer.weight.ndim == 2:
                return int(embedding_layer.weight.shape[1])
        return None

    def _infer_item_rec_size(self):
        if hasattr(self.rec_model, "item_embedding"):
            dim = self._embedding_dim(self.rec_model.item_embedding)
            if dim is not None:
                return dim
        if hasattr(self.rec_model, "item_embeddings"):
            dim = self._embedding_dim(self.rec_model.item_embeddings)
            if dim is not None:
                return dim
        return None

    def _infer_user_rec_size(self):
        if hasattr(self.rec_model, "user_embedding"):
            dim = self._embedding_dim(self.rec_model.user_embedding)
            if dim is not None:
                return dim
        if hasattr(self.rec_model, "user_embeddings"):
            dim = self._embedding_dim(self.rec_model.user_embeddings)
            if dim is not None:
                return dim
        return None

    def encode_items(self, seq):
        if self.hparams.rec_embed in ["SASRec", "ReCANet"]:
            item_rec_embs=self.rec_model.cacu_x(seq)
        elif self.hparams.rec_embed in ['Caser','GRU']:
            item_rec_embs=self.rec_model.item_embeddings(seq)
        else:
            raise ValueError(f"Unsupported rec_embed type: {self.hparams.rec_embed}")
        item_txt_embs=self.projector(item_rec_embs)
        return item_txt_embs

    def encode_users(self, user_ids):
        if self.user_projector is None:
            return None
        if hasattr(self.rec_model, "cacu_u"):
            user_rec_embs = self.rec_model.cacu_u(user_ids)
        elif hasattr(self.rec_model, "user_embeddings"):
            user_rec_embs = self.rec_model.user_embeddings(user_ids)
        elif hasattr(self.rec_model, "user_embedding"):
            user_rec_embs = self.rec_model.user_embedding(user_ids)
        else:
            if not self._warned_missing_user_embedding:
                self.print("User embedding requested but rec_model has no user embedding layer.")
                self._warned_missing_user_embedding = True
            return None
        if user_rec_embs.ndim == 3 and user_rec_embs.shape[1] == 1:
            user_rec_embs = user_rec_embs.squeeze(1)
        user_txt_embs = self.user_projector(user_rec_embs)
        if user_txt_embs.ndim == 3 and user_txt_embs.shape[1] == 1:
            user_txt_embs = user_txt_embs.squeeze(1)
        return user_txt_embs

    def embed_tokens(self, token_ids):
        embeds = self.llama_model.get_input_embeddings()(token_ids)
        return embeds

    def _build_attention_mask(self, tokens):
        """Return a usable attention mask even if the dataloader omitted one."""
        if hasattr(tokens, 'attention_mask') and tokens.attention_mask is not None:
            attention_mask = tokens.attention_mask.to(tokens.input_ids.device)
        else:
            attention_mask = None

        regenerated = tokens.input_ids.ne(self.llama_tokenizer.pad_token_id).long()
        if attention_mask is None:
            if not self._warned_missing_attention_mask:
                self.print('attention_mask missing in batch; regenerating from input_ids')
                self._warned_missing_attention_mask = True
            return regenerated

        needs_fix = attention_mask.sum(dim=-1) == 0
        if needs_fix.any():
            if not self._warned_missing_attention_mask:
                self.print('Found empty attention_mask rows; regenerating from input_ids')
                self._warned_missing_attention_mask = True
            attention_mask = attention_mask.clone()
            attention_mask[needs_fix] = regenerated[needs_fix]
        return attention_mask.to(regenerated.dtype)

    def wrap_emb(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)

        his_token_id=self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        item_token_id=self.llama_tokenizer("[ItemEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        user_token_id=self.llama_tokenizer("[UserEmb]", return_tensors="pt",add_special_tokens=False).input_ids.item()
        his_item_embeds= self.encode_items(batch["seq"])
        item_embeds=self.encode_items(batch["item_id"])
        user_embeds=self.encode_users(batch["user_id"]) if "user_id" in batch else None

        his_item_embeds = his_item_embeds.to(input_embeds.dtype)
        item_embeds = item_embeds.to(input_embeds.dtype)
        if user_embeds is not None:
            user_embeds = user_embeds.to(input_embeds.dtype)

        for i in range(len(batch["len_seq"])):
            if (batch["tokens"].input_ids[i]==his_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"].input_ids[i]==his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor,his_item_embeds[i,:batch["len_seq"][i].item()]):
                    input_embeds[i,idx]=item_emb
            if (batch["tokens"].input_ids[i]==item_token_id).nonzero().shape[0]>0:
                idx=(batch["tokens"].input_ids[i]==item_token_id).nonzero().item()
                input_embeds[i,idx]=item_embeds[i]
            if user_embeds is not None and (batch["tokens"].input_ids[i]==user_token_id).nonzero().shape[0]>0:
                idx_tensor=(batch["tokens"].input_ids[i]==user_token_id).nonzero().view(-1)
                for idx in idx_tensor:
                    input_embeds[i,idx]=user_embeds[i]
        return input_embeds

    def calculate_hr1(self,eval_content):
        correct_num=0
        valid_num=0
        total_num=0
        for i, generate_item in enumerate(eval_content["generate_item"]):
            real=eval_content["real_item"][i]
            total_num+=1
            generate=self._normalize_item_name(generate_item)
            real=self._normalize_item_name(real)
            if generate:
                valid_num+=1
                if real == generate:
                    correct_num+=1
        valid_ratio=valid_num/total_num if total_num else 0
        if valid_num>0:
            hr1=correct_num/valid_num
        else:
            hr1=0
        return valid_ratio,hr1

    def _extract_recommended_items(self, generated_text):
        """Parse recommended items strictly from `item:` field in model output."""
        text = str(generated_text).strip()
        candidates = []

        # Preferred format: item: {item_1, item_2}
        brace_match = re.search(r'item\s*[:：]\s*[\{｛]([^｝}]*)[\}｝]', text, flags=re.IGNORECASE | re.DOTALL)
        if brace_match:
            raw = brace_match.group(1)
            candidates.extend([x.strip() for x in re.split(r'[,、\n]+', raw) if x.strip()])
        else:
            line_match = re.search(r'item\s*[:：]\s*([^\n\r]+)', text, flags=re.IGNORECASE)
            if line_match:
                candidates.append(line_match.group(1).strip())

        normalized = []
        seen = set()
        for cand in candidates:
            name = self._normalize_item_name(cand)
            if name in self.all_item_name_set and name not in seen:
                normalized.append(name)
                seen.add(name)
        return normalized

    def _extract_item_and_explanation(self, generated_text):
        raw_text = str(generated_text).strip()
        items = self._extract_recommended_items(raw_text)
        generate_item = items[0] if items else ""

        explanation = ""
        explanation_match = re.search(r'explanation\s*[:：]\s*([^\n\r]+)', raw_text, flags=re.IGNORECASE)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        return generate_item, explanation

