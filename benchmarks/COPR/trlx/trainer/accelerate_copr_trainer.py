import json
import os
from dataclasses import is_dataclass
from time import time

import ray
import torch
import torch.nn.functional as F
import transformers
import trlx.utils.logging as logging
from ray.air import session
from transformers.tokenization_utils_base import BatchEncoding

from benchmarks.COPR.trlx.data.configs import TRLConfig
from benchmarks.COPR.trlx.models.modeling_copr import (
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
)
from benchmarks.COPR.trlx.pipeline.copr_pipeline import COPRPipeline
from benchmarks.COPR.trlx.trainer import register_trainer
from benchmarks.COPR.trlx.trainer.accelerate_base_trainer import (
    AccelerateRLTrainer,
)
from benchmarks.COPR.trlx.utils import filter_non_scalars
from benchmarks.COPR.trlx.utils.modeling import (
    flatten_dict,
    freeze_bottom_causal_layers,
    freeze_bottom_seq2seq_layers,
    get_delta_model_class,
    logprobs_of_labels,
    parse_delta_kwargs,
)

logger = logging.get_logger(__name__)


class COPRMiniBatchIterator:
    """A custom iterator for generating mini-batches from a PyTorch DataLoader."""

    def __init__(self, data_loader, mb_size, num_mb):
        """Initializes the MiniBatchIterator.

        Args:
            data_loader (torch.utils.data.DataLoader): The DataLoader to generate mini-batches from.
            mb_size (int): The size of each mini-batch.
            num_mb (int): The number of mini-batches to generate for each iteration.
        """
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        self.mb_size = mb_size
        self.num_mb = num_mb

    def __iter__(self):
        return self

    def __next__(self):  # noqa: C901
        # [bs*2,:]
        batch = next(self.data_loader_iter)
        if batch is None:
            logger.warning(
                'WARNING: Not enough samples to saturate the minibatch size. Increase the number '
                'of prompts or samples or decrease the minibatch size.'
            )
            raise StopIteration

        minibatches = []

        for mbi in range(self.num_mb):
            sliced_data = {}
            batch_dict = batch
            if is_dataclass(batch):
                batch_dict = batch.__dict__
            for key, value in batch_dict.items():
                assert value.shape[0] % 4 == 0
                bs = value.shape[0] // 4

                start_idx = mbi * self.mb_size
                end_idx = (mbi + 1) * self.mb_size
                sliced_data[key] = torch.cat(
                    [
                        value[start_idx:end_idx],
                        value[start_idx + bs : end_idx + bs],
                        value[start_idx + bs * 2 : end_idx + bs * 2],
                        value[start_idx + bs * 3 : end_idx + bs * 3],
                    ]
                )

                if self.num_mb > 1 and len(sliced_data[key]) == 0:
                    logger.warning(
                        'WARNING: MiniBatchIterator generated a minibatch with 0 elements. '
                        'This may be due to the wrong mb_size and/or num_mb or the last batch'
                        'in the dataset being smaller.'
                    )
                    sliced_data.pop(key)
                    break
                elif self.num_mb > 1 and len(sliced_data[key]) < self.mb_size:
                    logger.warning(
                        'WARNING: MiniBatchIterator generated a minibatch with fewer elements than mb_size. '
                        'This may be due to the wrong mb_size and/or num_mb or the last batch in the dataset '
                        'being smaller.'
                    )
            if not sliced_data:
                break

            if isinstance(batch, BatchEncoding):
                minibatch = BatchEncoding(sliced_data)
            elif is_dataclass(batch):
                minibatch = batch.__class__(**sliced_data)
            # else:
            #     minibatch = sliced_data

            minibatches.append(minibatch)

        if not minibatches:
            raise StopIteration

        return minibatches


def logprobs_of_labels_fp32(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits.
    """
    logprobs = F.log_softmax(logits.float(), dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).half()
    return logprobs_labels.squeeze(-1)


def get_tensor_stats(xs: torch.Tensor):
    if xs.numel() == 0:
        return dict(mean=0, min=0, max=0, std=0)
    return dict(
        mean=xs.mean().item(),
        min=xs.min().item(),
        max=xs.max().item(),
        std=torch.sqrt((xs - xs.mean()).pow(2).mean()),
    )


@register_trainer
class AccelerateCOPRTrainer(AccelerateRLTrainer):
    """COPR Accelerate Trainer"""

    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self.config = config
        # self.lm_loss_fn=torch.nn.CrossEntropyLoss(ignore_index=2)
        self.lm_loss_fn = torch.nn.CrossEntropyLoss()
        self.log_lambda = torch.tensor([self.config.method.log_lambda])
        self.ref_model = self.get_arch(self.config)
        self.ref_model.to(self.accelerator.device)
        self.ref_model.eval()

    def setup_model(self):
        """Returns a model derived from an instance's TRLConfig"""
        logger.info(f'Initializing model: {self.config.model.model_path}')

        # Retrieves model equipped for ppo, ilql, etc
        model = self.get_arch(self.config)
        if self.config.model.model_arch_type == 'seq2seq':
            freeze_bottom_seq2seq_layers(
                model.base_model, self.config.model.num_layers_unfrozen
            )
        else:
            freeze_bottom_causal_layers(
                model.base_model, self.config.model.num_layers_unfrozen
            )
        # Set the delta tuning strategies
        if self.config.model.delta_kwargs is not None:
            delta_type, delta_kwargs = parse_delta_kwargs(
                model.base_model.config,
                self.config.model.delta_kwargs,
                self.config.model.num_layers_unfrozen,
            )
            delta_model_class = get_delta_model_class(delta_type)
            delta_model = delta_model_class(model.base_model, **delta_kwargs)
            delta_model.freeze_module(exclude=['deltas'], set_state_dict=True)
            if self.accelerator.is_main_process:
                delta_model.log()

        return model

    def get_arch(self, config: TRLConfig):
        """Get the model"""
        model_class = AutoModelForCausalLMWithHydraValueHead
        if config.model.model_arch_type == 'seq2seq':
            model_class = AutoModelForSeq2SeqLMWithHydraValueHead

        from_fn = model_class.from_pretrained
        # backward-compat: Try to create a randomly initialized architecture from a config
        if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
            from_fn = model_class.from_config

        return from_fn(
            config.model.model_path,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
            cache_dir=config.model.cache_dir,
        )

    def loss(self, batch):
        """Batch structure,
        bs: batch size,
        L: sequence length
        |-------------------------|
        | chosen (new task): bs*L |
        |-------------------------|
        | chosen (old task): bs*L |
        |-------------------------|
        | reject (new task): bs*L |
        |-------------------------|
        | reject (old task): bs*L |
        |-------------------------|

        """
        # paddding_side: left
        starts = batch['s_res'] - 1
        end = -1

        n_samples = batch['input_ids'].shape[0]
        n_queries = n_samples // 2
        n_new = n_queries // 2
        assert (
            n_samples == n_queries * 2
        ), f'not equal: n_samples={n_samples},n_queries={n_queries}'

        # policy model forward()
        outputs = self.model(
            batch['input_ids'], batch['attention_mask'], return_dict=True
        )
        logits = outputs.logits
        # [B,L]
        value = outputs.value

        # print(f"Values after forward: {value}")

        ## 不使用真实标签，按照value对数据进行排序
        # 重排id
        re_ids = list(range(n_samples))

        if '[unlabel]' in self.config.method.rdpo_type:
            ## RM-LOSS必须无效
            self.config.method.coef_rm = 0.0
            assert self.config.method.coef_rm == 0.0
            for ix in range(n_queries):
                # 正样本得分小于负样本 -> 交换
                if (
                    value[ix, starts[ix] :].mean()
                    < value[ix + n_queries, starts[ix + n_queries] :].mean()
                ):
                    tmp = re_ids[ix]
                    re_ids[ix] = re_ids[ix + n_queries]
                    re_ids[ix + n_queries] = tmp
            # print(re_ids)
            batch['attention_mask'] = batch['attention_mask'][re_ids]
            batch['input_ids'] = batch['input_ids'][re_ids]
            batch['s_res'] = batch['s_res'][re_ids]
            batch['s_pmt'] = batch['s_pmt'][re_ids]

            logits = logits[re_ids]
            value = value[re_ids]

        # reference model forward()
        with torch.no_grad():
            if hasattr(self.model, 'frozen_head'):
                ref_logits = self.model.forward_hydra(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_dict=True,
                ).logits
            else:
                ref_logits = self.ref_model(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_dict=True,
                ).logits
        ref_logits = ref_logits.to(batch['input_ids'].device)
        # compute the log probablity of all label tokens
        logprobs = logprobs_of_labels(logits[:, :-1, :], batch['input_ids'][:, 1:])
        # print(f"log probs: {logprobs}")
        # assert log probs doesnt have nan
        assert not torch.isnan(logprobs).any(), 'logprobs has nan'
        ref_logprobs = logprobs_of_labels(
            ref_logits[:, :-1, :], batch['input_ids'][:, 1:]
        )
        # print(f"ref log probs: {ref_logprobs}")
        # assert ref log probs doesnt have nan
        assert not torch.isnan(ref_logprobs).any(), 'ref logprobs has nan'
        N_vocab = logits.shape[-1]

        # [2 * n_queries ,] This is pi(y|x)
        logprobs_response = []
        ref_logprobs_response = []
        avg_values = []

        LM_loss_w = 0.0
        LM_loss_l = 0.0

        for ix in range(n_samples):
            # print(f"logprobs shape: {logprobs.shape}, starts[{ix}]: {starts[ix]}")
            slice_tensor = logprobs[ix, starts[ix] :]
            # print(f"slice_tensor shape: {slice_tensor.shape}")
            if slice_tensor.numel() == 0:
                print('Empty slice detected')
            logp = slice_tensor.mean().unsqueeze(dim=0)
            rlogp = ref_logprobs[ix, starts[ix] :].mean().unsqueeze(dim=0)
            avg_value = value[ix, starts[ix] :].mean().unsqueeze(dim=0)

            logprobs_response.append(logp)
            ref_logprobs_response.append(rlogp)
            avg_values.append(avg_value)
            # print(ix, starts[ix], end, logprobs.shape, batch['input_ids'].shape, logits.shape ,logprobs[ix, starts[ix]: end], logprobs[ix, starts[ix]: end].mean(), logp, rlogp )

            # LM loss
            if ix < n_queries:
                # print(logits[ix, starts[ix]: end, :].shape,N_vocab, batch['input_ids'][ix, starts[ix] + 1:].shape)
                LM_loss_w += (
                    1
                    / n_queries
                    * self.lm_loss_fn(
                        logits[ix, starts[ix] : end, :].view([-1, N_vocab]),
                        batch['input_ids'][ix, starts[ix] + 1 :].view(-1),
                    )
                )
            else:
                LM_loss_l += (
                    1
                    / n_queries
                    * self.lm_loss_fn(
                        logits[ix, starts[ix] : end, :].view([-1, N_vocab]),
                        batch['input_ids'][ix, starts[ix] + 1 :].view(-1),
                    )
                )
        # print(f"log probs before cat: {logprobs_response}")
        logprobs_response = torch.cat(logprobs_response).to(batch['input_ids'].device)
        ref_logprobs_response = torch.cat(ref_logprobs_response).to(
            batch['input_ids'].device
        )
        avg_values = torch.cat(avg_values).to(batch['input_ids'].device)

        ## [n_queries, ] log (pi(y|x)/pi_ref(y|x)) = log pi(y|x) - log pi_ref(y|x)
        ratio_w = logprobs_response[:n_queries] - ref_logprobs_response[:n_queries]
        ratio_l = logprobs_response[n_queries:] - ref_logprobs_response[n_queries:]

        dpo_sigma = torch.sigmoid(ratio_l - ratio_w)

        ## rm loss
        if self.config.method.margin_rm == 0.0:
            rm_clip = avg_values[:n_queries] - avg_values[n_queries:]
        else:
            rm_clip = torch.clamp(
                -self.config.method.margin_rm
                + avg_values[:n_queries]
                - avg_values[n_queries:],
                -self.config.method.cliprange_reward,
                0.0,
            )

        ## RM训练loss与acc
        RW_loss = -torch.log(torch.sigmoid(rm_clip)).mean()
        rm_acc = sum(
            avg_values[re_ids[:n_queries]] > avg_values[re_ids[n_queries:]]
        ) / (avg_values.shape[0] / 2)

        # hyper-parameter
        beta = self.config.method.dpo_beta

        if '[gaussian_reward]' in self.config.method.rdpo_type:
            rm_score = torch.randn(
                avg_values.shape, dtype=avg_values.dtype, device=avg_values.device
            ).view([2, -1])
            rm_score = rm_score.sort(dim=0)[0].view(-1)
            rm_score = rm_score * (2**0.5)
        elif '[linear_reward]' in self.config.method.rdpo_type:
            rm_score = torch.ones_like(avg_values) * self.config.method.margin_rm
            rm_score[n_queries:] = -1.0 * self.config.method.margin_rm
        else:
            NotImplementedError(
                f"rdpo_type='{self.config.method.rdpo_type}' is not implemented"
            )

        ## 初始化配分函数Z与最优解
        Z = torch.zeros([1])
        optimal_sol = torch.zeros([1])
        # print(f"log probs response: {logprobs_response}")

        scale_optimal = (1 / beta * rm_score.detach()).exp()
        P_policy_star = (
            logprobs_response.detach().exp()
            * scale_optimal
            / (logprobs_response.detach().exp() * scale_optimal).sum()
        )
        P_policy_star = P_policy_star.clamp(min=1e-9)
        P_policy = logprobs_response.exp() / logprobs_response.exp().sum()
        P_policy = P_policy.clamp(min=1e-9)
        # print(f"P_policy_star: {P_policy_star}")
        # print(f"P_policy: {P_policy}")
        # fit for new samples
        RDPO_loss = (
            P_policy[:n_new].log() - P_policy_star[:n_new].log()
        ).square().sum() + (
            P_policy[2 * n_new : 3 * n_new].log()
            - P_policy_star[2 * n_new : 3 * n_new].log()
        ).square().sum()
        # print(f"RDPO_loss: {RDPO_loss}")
        RDPO_loss = RDPO_loss / n_queries

        ## reg for old samples
        reg_loss = (
            logprobs_response[n_new : 2 * n_new]
            - ref_logprobs_response[n_new : 2 * n_new]
        ).square().sum() + (
            logprobs_response[3 * n_new : 4 * n_new]
            - ref_logprobs_response[3 * n_new : 4 * n_new]
        ).square().sum()
        reg_loss = reg_loss / n_queries
        if reg_loss.item() == 0.0:
            # introduce a small value to avoid nan
            reg_loss = torch.tensor(1e-9, device=reg_loss.device)

        # print(f"reg loss: {reg_loss}")
        assert not torch.isnan(RDPO_loss).any(), 'RDPO_loss has nan'
        assert not torch.isnan(RW_loss).any(), 'RW_loss has nan'
        assert not torch.isnan(reg_loss).any(), 'reg_loss has nan'
        assert not torch.isnan(LM_loss_w).any(), 'LM_loss_w has nan'
        assert not torch.isnan(LM_loss_l).any(), 'LM_loss_l has nan'

        Jc = (reg_loss - self.config.method.constraint_threshold).cpu().item()

        ## Lagrangian method for updating lambda
        # log lambda(k+1) = log lambda(k+1) + alpha* lambda(k+1) * Jc
        self.log_lambda = (
            self.log_lambda + self.config.method.lambda_lr * self.log_lambda.exp() * Jc
        )

        new_lambda = self.log_lambda.exp().cpu().item()
        # print(f"new lambda: {new_lambda}")

        ## Lagrangian method for updating theta
        loss = (
            self.config.method.coef_rm
            / (new_lambda + self.config.method.coef_rm)
            * RW_loss
            + self.config.method.coef_nll * LM_loss_w
            + self.config.method.coef_dpo * RDPO_loss
            + new_lambda / (new_lambda + self.config.method.coef_rm) * reg_loss
        )

        # print(f"loss: {loss}")
        assert not torch.isnan(loss).any(), 'loss has nan'

        ## wandb
        stats = dict(
            losses=dict(
                total_loss=loss.item(),
                RDPO_loss=RDPO_loss.item(),
                RW_loss=RW_loss.item(),
                reg_loss=reg_loss.item(),
                LM_loss_w=LM_loss_w.item(),
                LM_loss_l=LM_loss_l.item(),
            ),
            RM_status=dict(
                rm_acc=rm_acc.item(),
                socre_w=get_tensor_stats(avg_values[:n_queries]),
                socre_l=get_tensor_stats(avg_values[n_queries:]),
            ),
            RDPO_status=dict(
                rm_score=get_tensor_stats(rm_score),
                Z=get_tensor_stats(Z),
                optimal_sol=get_tensor_stats(optimal_sol),
            ),
            DPO_status=dict(
                dpo_sigma=get_tensor_stats(dpo_sigma),
                ratio_w=get_tensor_stats(ratio_w),
                ratio_l=get_tensor_stats(ratio_l),
                logprobs_w=get_tensor_stats(logprobs_response[:n_queries]),
                ref_logprobs_w=get_tensor_stats(ref_logprobs_response[:n_queries]),
                logprobs_l=get_tensor_stats(logprobs_response[n_queries:]),
                ref_logprobs_l=get_tensor_stats(ref_logprobs_response[n_queries:]),
            ),
            Lagrangian=dict(
                lagrangian_multiplier=new_lambda,
            ),
        )

        return loss, flatten_dict(stats)

    def prepare_learning(self):
        train_dataloader = self.store.create_loader(self.config.train.batch_size)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        (
            self.model,
            self.opt,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.model, self.opt, train_dataloader, eval_dataloader
        )

        self.n_updates_per_batch = 1
        self.total_steps = self.config.train.epochs * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def make_experience(self, samples, input_length, output_length):
        self.store = COPRPipeline(samples, input_length, output_length, self.tokenizer)

    def learn(self):  # noqa: C901
        """Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`"""
        logger.info('Starting RDPO training')

        self.prepare_learning()
        self.iter_count = 0
        self.nth_evaluation = 0

        if ray.is_initialized():
            checkpoint = session.get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as dir:
                    self.accelerator.load_state(dir)

                    with open(os.path.join(dir, 'state.json')) as f:
                        state = json.load(f)
                        self.iter_count = state['iter_count']
        else:
            results = self.evaluate()
            self.accelerator.log(results, step=self.iter_count)

        tbar = logging.tqdm(
            initial=self.iter_count,
            total=self.total_steps,
            disable=not self.accelerator.is_local_main_process,
            position=0,
            leave=True,
        )

        best_reward = -float('inf')

        # For each epoch
        for _ in range(self.config.train.epochs):
            # For each batch
            for mbs in COPRMiniBatchIterator(
                self.train_dataloader, self.mb_size, self.num_mb
            ):
                # For each update per batch
                for _ in range(self.n_updates_per_batch):
                    # Note that whereas standard policy gradient methods perform one
                    # gradient update per batch, PPO for example commonly performs
                    # multiple gradient updates on the same batch of data.
                    # https://arxiv.org/pdf/1707.06347.pdf
                    forward_time = 0
                    backward_time = 0
                    stats_accum = []
                    for mb in mbs:
                        with self._accumulate():
                            forward_time -= time()
                            loss, stats = self.loss(mb)
                            forward_time += time()
                            backward_time -= time()
                            self.accelerator.backward(loss)
                            backward_time += time()
                            stats_accum.append(stats)

                    forward_time /= self.num_mb
                    backward_time /= self.num_mb
                    # TODO(Dahoas): Best way to combine stats between mbs?
                    # How does accelerate do it?
                    stats = {
                        key: sum([stats[key] for stats in stats_accum]) / self.num_mb
                        for key in stats_accum[0]
                    }

                    self.opt.step()
                    self.opt.zero_grad()
                    self.scheduler.step()
                    self.iter_count += 1

                    if (
                        self.iter_count % self.config.train.checkpoint_interval == 0
                        or self.iter_count >= self.total_steps
                    ):
                        subfolder = f'checkpoint_{self.iter_count:0{len(str(self.total_steps))}d}'
                        directory = os.path.join(
                            self.config.train.checkpoint_dir, subfolder
                        )
                        logger.info(f'Saving intermediate checkpoint into {directory}')
                        if self.config.train.save_optimizer:
                            self.save(directory)
                        else:
                            self.save_pretrained(directory)

                    stats['time/forward'] = forward_time
                    stats['time/backward'] = backward_time
                    for group_number, lr in enumerate(self.scheduler.get_last_lr()):
                        stats[f'learning_rate_group_{group_number}'] = lr

                    if (
                        self.iter_count % self.config.train.eval_interval == 0
                        or self.iter_count >= self.total_steps
                    ):
                        results = self.evaluate()
                        stats.update(results)
                        if ray.is_initialized():
                            session.report(
                                filter_non_scalars(stats), checkpoint=checkpoint
                            )

                        # always save checkpoint with the greatest mean reward
                        if self.config.train.save_best:
                            if stats.get('reward/mean', -float('inf')) > best_reward:
                                best_reward = stats.get('reward/mean')
                                do_save = True
                            # in case ILQL reports reward estimate as one of its metrics
                            elif (
                                stats.get('metrics/reward', -float('inf')) > best_reward
                            ):
                                best_reward = stats.get('metrics/reward')
                                do_save = True
                            else:
                                do_save = False
                            do_save = torch.tensor(
                                do_save, device=self.accelerator.device
                            )
                            if torch.distributed.is_initialized():
                                torch.distributed.all_reduce(
                                    do_save, torch.distributed.ReduceOp.MAX
                                )
                            if do_save:
                                directory = os.path.join(
                                    self.config.train.checkpoint_dir, 'best_checkpoint'
                                )
                                logger.info(
                                    f'Saving the best state so far into {directory}'
                                )
                                if self.config.train.save_optimizer:
                                    self.save(directory)
                                else:
                                    self.save_pretrained(directory)

                    desc = ' | '.join(
                        f'{k}: {v:.2f}'
                        for k, v in stats.items()
                        if k.startswith('loss')
                    )
                    tbar.set_description(f'[{desc}]')
                    tbar.update()

                    self.accelerator.log(stats, step=self.iter_count)

                    if self.iter_count >= self.total_steps:
                        return results

                self.post_backward_callback()

            self.post_epoch_callback()
        tbar.close()
