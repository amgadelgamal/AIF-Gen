import json
import os
import pathlib
import random
from typing import List

import torch

# from datasets import load_dataset
from transformers import AutoTokenizer

from benchmarks.COPR.trlx import trlx
from benchmarks.COPR.trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from benchmarks.COPR.trlx.models.modeling_copr import COPRConfig
from benchmarks.COPR.trlx.trainer import register_trainer
from benchmarks.COPR.trlx.trainer.accelerate_copr_trainer import (
    AccelerateCOPRTrainer,
)

register_trainer(AccelerateCOPRTrainer)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
train_bs_1gpu = 2
rollout_times = 1000
num_rollouts = 16
N_ddp = 1
ppo_epochs = 1
chunk_size = 8
num_layers_unfrozen = -1

README = 'COPR'
# get the SCRATCH folder from var $SCRATCH
model_cache_dir = os.environ['SCRATCH'] + 'TRLX/model_cache'  # SCRATCH For Mila
modeL_cache_dir = pathlib.Path(model_cache_dir)
checkpoint_dir = os.environ['SCRATCH'] + '/TRLX/COPR/checkpoints'  # SCRATCH For Mila
SFT_MODEL_PATH = 'meta-llama/Llama-3.2-1B'  # could be HF or local
DATA_DIR = 'benchmarks/COPR'  # where there are json files

print(f'total batch:\t{int(train_bs_1gpu * N_ddp)}')
print(f'num_layers_unfrozen:\t{num_layers_unfrozen}')
print(f'SFT_MODEL_PATH:\t{SFT_MODEL_PATH}')
print(f'DATA_DIR:\t{DATA_DIR}')
print(README)

GROUP_NAME = 'Alignment'

config = TRLConfig(
    train=TrainConfig(
        seq_length=2048,
        epochs=rollout_times,
        total_steps=1000000,
        batch_size=train_bs_1gpu,
        checkpoint_interval=2000,
        checkpoint_dir=checkpoint_dir,
        eval_interval=200,
        pipeline='COPRPipeline',
        trainer='AccelerateCOPRTrainer',
        group_name=GROUP_NAME,
    ),
    model=ModelConfig(
        model_path=SFT_MODEL_PATH, num_layers_unfrozen=num_layers_unfrozen
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path=SFT_MODEL_PATH, truncation_side='left', padding_side='left'
    ),
    optimizer=OptimizerConfig(
        name='adamw',
        kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6),
    ),
    scheduler=SchedulerConfig(
        name='cosine_annealing', kwargs=dict(T_max=10000, eta_min=8e-6)
    ),
    method=COPRConfig(
        name='COPRConfig',
        num_rollouts=num_rollouts,
        chunk_size=chunk_size,
        ppo_epochs=ppo_epochs,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward='running',
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10.0,
        gen_kwargs=dict(
            max_new_tokens=256,
            top_k=20,
            top_p=1.0,
            do_sample=True,
        ),
        cache_dir=model_cache_dir,
        dpo_beta=0.8,
        coef_rm=0.0001,
        coef_nll=0.8,
        coef_dpo=0.8,
        coef_reg=0.8,
        margin_rm=0.8,
        margin_dpo=0.8,
        log_lambda=0.0001,
        lambda_lr=0.0001,
        constraint_threshold=0.1,
        rdpo_type='[score_reward]_[ref_rdpo]_[unlabel]_[gaussian_reward]',
    ),
    cache_dir=model_cache_dir,
)


def get_score_local(prompts, predictions):
    # TODO: Implement the scoring function based on judge models
    return [0] * len(prompts)


def get_scores(prompts, predictions):
    scores_list = get_score_local(prompts, predictions)
    scores = torch.tensor(scores_list)
    return scores


def reward_fn(samples: List[str], prompts=None, outputs=None, **kwargs):
    prompts = [text for text in prompts]
    predictions = [text for text in outputs]
    scores = get_scores(prompts, predictions)
    return scores


def get_samples(data_dir):
    json_list = os.listdir(data_dir)
    json_list = [f for f in json_list if f.endswith('.json')]
    all_data = []
    for file in json_list:
        with open(f'{data_dir}/{file}', 'r', encoding='utf-8') as f:
            data_list = json.loads(f.read())
            all_data.extend(data_list)

    def construct_pairs(sample):
        item_list = []
        prompt = sample['prompt']
        ans_list = sample['answers']
        replay_prompt = sample.get('replay_prompt', prompt)
        replay_ans_list = sample.get('replay_answers', ans_list)
        n = len(ans_list)

        for i in range(n):
            for j in range(i + 1, n):
                if ans_list[i]['score'] > ans_list[j]['score']:
                    item = {
                        'prompt': prompt,
                        'chosen': ans_list[i]['answer'],
                        'rejected': ans_list[j]['answer'],
                        'score_c': ans_list[i]['score'],
                        'score_r': ans_list[j]['score'],
                        'replay_prompt': replay_prompt,
                        'replay_chosen': replay_ans_list[i]['answer'],
                        'replay_rejected': replay_ans_list[j]['answer'],
                    }
                    item_list.append(item)
        return item_list

    all_pairs = []
    for sample in all_data:
        pair_list = construct_pairs(sample)
        all_pairs.extend(pair_list)
    random.seed(666)
    random.shuffle(all_pairs)
    n = len(all_pairs)
    n1 = int(0.8 * n)
    n2 = int(0.9 * n)
    train = all_pairs[:n1]
    valid = all_pairs[n1:n2]
    test = all_pairs[n2:]
    return train, valid, test


if __name__ == '__main__':
    # if 'llama' in config.tokenizer.tokenizer_path.lower():
    #     tokenizer = LlamaTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    # else: dont need this for now - the code works with llama
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    max_length_input = (
        config.train.seq_length - config.method.gen_kwargs['max_new_tokens']
    )

    train_samples, valid_samples, _ = get_samples(DATA_DIR)

    trainer = trlx.train(
        metric_fn=lambda **kwargs: {'reward': reward_fn(**kwargs)},
        samples=train_samples,
        eval_prompts=[sample['prompt'] for sample in valid_samples],
        config=config,
    )
