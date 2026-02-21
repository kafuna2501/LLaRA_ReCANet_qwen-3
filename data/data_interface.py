import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import random
import torch
import argparse
import os



class TrainCollater:
    def __init__(self,
                 prompt_list=None,
                 llm_tokenizer=None,
                 system_prompt="",
                 use_chat_template=False,
                 train=False,
                 terminator="\n",
                 max_step=1):
        self.prompt_list = prompt_list
        self.llm_tokenizer = llm_tokenizer
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template
        self.train=train
        self.terminator = terminator
        self.max_step = max_step
        self.cur_step = 1

    def _render_prompt(self, user_text):
        if not self.use_chat_template:
            return user_text

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_text})
        try:
            return self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            if self.system_prompt:
                return f"System: {self.system_prompt}\nUser: {user_text}\nAssistant:"
            return f"User: {user_text}\nAssistant:"

    def __call__(self, batch):
        if isinstance(self.prompt_list,list):
            instruction = random.choice(self.prompt_list)
            inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)
        else:
            instruction = sample["instruction_input"] if "instruction_input" in sample else None
            inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)

        thresh_hold = self.cur_step/self.max_step
        p = random.random()
        if p < thresh_hold or not self.train:
            for i, sample in enumerate(batch):
                input_text=inputs_text[i]
                if '[TodayHere]' in input_text:
                    today = sample['date']     # これが cur_date に相当
                    input_text = input_text.replace('[TodayHere]', str(today))
                if '[UserIdHere]' in input_text:
                    input_text = input_text.replace('[UserIdHere]', str(sample['user_id']))
                if '[TodaySeqHere]' in input_text:
                    insert_prompt=", ".join([seq_title+' [HistoryEmb]' for seq_title in sample['seq_name']])
                    input_text=input_text.replace('[TodaySeqHere]',insert_prompt)
                if '[HistoryHere]' in input_text:
                    insert_prompt=", ".join([t + ' [HistoryEmb]' for t in sample['hist_seq_name']])
                    input_text=input_text.replace('[HistoryHere]',insert_prompt)
                inputs_text[i]=input_text
            flag = False
        else:
            for i, sample in enumerate(batch):
                input_text=inputs_text[i]
                if '[TodayHere]' in input_text:
                    today = sample['date']     # これが cur_date に相当
                    input_text = input_text.replace('[TodayHere]', str(today))
                if '[UserIdHere]' in input_text:
                    input_text = input_text.replace('[UserIdHere]', str(sample['user_id']))
                if '[TodaySeqHere]' in input_text:
                    insert_prompt=", ".join([seq_title+' [PH]' for seq_title in sample['seq_name']])
                    input_text=input_text.replace('[TodaySeqHere]',insert_prompt)
                if '[HistoryHere]' in input_text:
                    insert_prompt=", ".join([t + ' [HistoryEmb]' for t in sample['hist_seq_name']])
                    input_text=input_text.replace('[HistoryHere]',insert_prompt)
                inputs_text[i]=input_text
            flag = True
        self.cur_step += 1

        inputs_text = [self._render_prompt(x) for x in inputs_text]

        targets_text = [
            "item: {" + str(sample['correct_answer']) + "}\n"
            "explanation: Based on this customer's recent purchase pattern."
            for sample in batch
        ]

        if self.train:
            targets_text=[target_text+self.terminator for target_text in targets_text]
            inputs_pair = [[p, t] for p, t in zip(inputs_text, targets_text)]
            batch_tokens = self.llm_tokenizer(
                inputs_pair,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True)
            new_batch={"tokens":batch_tokens,
                       "seq": torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                       "user_id": torch.stack([torch.tensor(sample['user_id']) for sample in batch], dim=0),
                       "len_seq": torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                       "item_id": torch.stack([torch.tensor(sample['item_id']) for sample in batch], dim=0),
                       "flag":flag,
                       }
        else:
            batch_tokens = self.llm_tokenizer(
                inputs_text,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True)
            new_batch={"tokens":batch_tokens,
                       "seq": torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                       "user_id": torch.stack([torch.tensor(sample['user_id']) for sample in batch], dim=0),
                       "len_seq": torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                       "item_id": torch.stack([torch.tensor(sample['item_id']) for sample in batch], dim=0),
                       "correct_answer": targets_text,
                       }
        return new_batch

class DInterface(pl.LightningDataModule):

    def __init__(self,
                 llm_tokenizer=None,
                 num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.llm_tokenizer=llm_tokenizer
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.load_data_module()
        self.load_prompt(kwargs['prompt_path'])

        self.trainset = self.instancialize(stage='train')
        self.valset = self.instancialize(stage='val')
        self.testset = self.instancialize(stage='test')
        self.max_steps = self.max_epochs*(len(self.trainset)//self.batch_size)//self.num_workers

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=TrainCollater(
                              prompt_list=self.prompt_list,
                              llm_tokenizer=self.llm_tokenizer,
                              system_prompt=self.kwargs.get('system_prompt', ''),
                              use_chat_template=self.kwargs.get('use_chat_template', False),
                              train=True,
                              max_step=self.max_steps,
                          ))

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=TrainCollater(
                              prompt_list=self.prompt_list,
                              llm_tokenizer=self.llm_tokenizer,
                              system_prompt=self.kwargs.get('system_prompt', ''),
                              use_chat_template=self.kwargs.get('use_chat_template', False),
                              train=False,
                          ))

    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=TrainCollater(
                              prompt_list=self.prompt_list,
                              llm_tokenizer=self.llm_tokenizer,
                              system_prompt=self.kwargs.get('system_prompt', ''),
                              use_chat_template=self.kwargs.get('use_chat_template', False),
                              train=False,
                          ))

    def load_data_module(self):
        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        sig = inspect.signature(self.data_module.__init__)
        class_args = list(sig.parameters.keys())[1:]   # self を除外

        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)

    def load_prompt(self,prompt_path):
        if os.path.isfile(prompt_path):
            with open(prompt_path, 'r') as f:
                raw_prompt_text = f.read()
            raw_prompt_text = raw_prompt_text.strip()
            if not raw_prompt_text:
                self.prompt_list = []
            elif "\n<|PROMPT_SPLIT|>\n" in raw_prompt_text:
                self.prompt_list = [p.strip() for p in raw_prompt_text.split("\n<|PROMPT_SPLIT|>\n") if p.strip()]
            else:
                # Treat the whole file as a single prompt so multi-line prompts are preserved.
                self.prompt_list = [raw_prompt_text]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []
