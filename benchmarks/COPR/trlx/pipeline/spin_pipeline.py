import json
import os
import time
from typing import Any, Dict, Iterable, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from benchmarks.COPR.trlx.data.spin_types import SPINBatch, SPINElement
from benchmarks.COPR.trlx.pipeline import (
    BasePipeline,
    BaseRolloutStore,
    register_datapipeline,
)


class SPINRolloutStorage(BaseRolloutStore):
    """Rollout storage for training SPIN."""

    def __init__(self, pad_token_id, padding_side):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.history: Iterable[SPINElement] = [None]

    def push(self, exps: Iterable[SPINElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str):
        assert os.path.exists(location)

        fpath = os.path.join(location, f'epoch-{str(time.time())}.json')

        def exp_to_dict(exp):
            return {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.history]
        with open(fpath, 'w') as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> SPINElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[SPINElement]):
            if self.padding_side == 'right':
                # Right padding of already right-padded queries
                query_tensors = pad_sequence(
                    [elem.query_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                )
            else:
                # Left padding of already left-padded queries
                query_tensors = pad_sequence(
                    [elem.query_tensor.flip(0) for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ).flip(1)

            return SPINBatch(
                query_tensors,
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.values for elem in elems], padding_value=0.0, batch_first=True
                ),
                torch.tensor([elem.old_logp_response for elem in elems]),
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)


@register_datapipeline
class SPINPipeline(BasePipeline):
    """Dataloader which is used to supply prompts for either training or evaluation.

    Args:
        prompts (`List[str]` or `List[Dict[str, Any]]`): list of raw text prompts or a dictionary with a required
            key `"prompt"` and extra information, that would be passed along the generation for that prompt as a
            keyword argument to a reward function.
        max_prompt_length (`int`): max length of the prompt, if exceeded the prompt will be truncated according to
            tokenizer's truncation setting.
        tokenizer (`transformers.PreTrainedTokenizer`): a tokenizer to tokenize prompts with.
        add_special_tokens (`bool`): whether to encode prompts with tokenizer's special tokens (passed directly
            into `tokenizer.encode`)
    """

    def __init__(
        self,
        prompts: Union[Dict[str, Any], List[str]],
        labels: Union[Dict[str, Any], List[str]],
        max_prompt_length: int,
        max_label_length: int,
        tokenizer: PreTrainedTokenizer,
        add_special_tokens: bool = False,
    ):
        super().__init__()

        if isinstance(prompts[0], dict):
            metadata = prompts
            prompts = [x.pop('prompt') for x in metadata]
        else:
            metadata = [{}] * len(prompts)

        assert tokenizer.truncation_side == 'left'
        assert tokenizer.padding_side == 'left'

        # for generate 应该左截断
        model_inputs_x = tokenizer(
            prompts,
            truncation=True,
            padding=False,
            max_length=max_prompt_length,
            add_special_tokens=add_special_tokens,
        )
        # for p(y|x) # 应该右截断
        tokenizer.truncation_side = 'right'
        model_inputs_y = tokenizer(
            labels,
            truncation=True,
            padding=False,
            max_length=max_label_length,
            add_special_tokens=False,
        )
        tokenizer.truncation_side = 'left'
        ######################################

        x_tokens = model_inputs_x['input_ids']
        x_masks = model_inputs_x['attention_mask']

        y_tokens = model_inputs_y['input_ids']
        y_masks = model_inputs_y['attention_mask']

        self.tokenizer = tokenizer

        self.prompts = [
            {
                'x_tokens': x_token,
                'x_masks': x_mask,
                'y_tokens': y_token,
                'y_masks': y_mask,
                'y_len': len(y_token),
                **metadata,
            }
            for x_token, x_mask, y_token, y_mask, metadata in zip(
                x_tokens, x_masks, y_tokens, y_masks, metadata
            )
        ]

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        def collate_fn(xs):
            out = self.tokenizer.pad(
                [{'input_ids': x['x_tokens']} for x in xs], return_tensors='pt'
            )

            # labels = list(map(torch.LongTensor, [ x["y_tokens"] for x in xs] ))
            # maxsize = max(map(len, labels))
            # labels = [
            #     F.pad(
            #         output,
            #         (0, maxsize - len(output)), # 左侧补0个[PAD], 右侧补maxsize-len(output)个[PAD]
            #         value=self.tokenizer.pad_token_id,
            #     )
            #     for output in labels
            # ]
            # labels = torch.vstack(labels)
            # # chosen as gold
            # out['label_ids'] = labels

            out['label_ids'] = [x['y_tokens'] for x in xs]
            out['label_len'] = torch.LongTensor([x['y_len'] for x in xs])

            for key in xs[0]:
                if key not in ['x_tokens', 'x_masks', 'y_tokens', 'y_masks', 'y_len']:
                    out[key] = [x[key] for x in xs]
            return out

        return DataLoader(
            self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
        )


# ###################### debug ######################
# if __name__ == '__main__':
#     from transformers import AutoTokenizer
#     import torch
#     import torch.nn.functional as F
#     tokenizer = AutoTokenizer.from_pretrained("D:\work\Research_HUB\RLHF\\trlx\examples\cl_rlhf\download\Llama-2-7b-hf")
#     data = torch.load("D:\work\Research_HUB\RLHF\\trlx\examples\cl_rlhf\data_dir\TILv3\\task-0\\trainvaltest.pt")
