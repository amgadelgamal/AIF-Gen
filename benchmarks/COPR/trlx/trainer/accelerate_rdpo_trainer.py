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
from benchmarks.COPR.trlx.models.modeling_rdpo import (
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
)
from benchmarks.COPR.trlx.pipeline.dvpo_pipeline import DVPOPipeline
from benchmarks.COPR.trlx.trainer import register_trainer
from benchmarks.COPR.trlx.trainer.accelerate_base_trainer import (
    AccelerateRLTrainer,
)
from benchmarks.COPR.trlx.utils import filter_non_scalars
from benchmarks.COPR.trlx.utils.modeling import (
    flatten_dict,
    freeze_bottom_causal_layers,
    freeze_bottom_seq2seq_layers,
    freeze_emb_causal_layers,
    get_delta_model_class,
    logprobs_of_labels,
    parse_delta_kwargs,
)

logger = logging.get_logger(__name__)


class RDPOMiniBatchIterator:
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
                assert value.shape[0] % 2 == 0
                bs = value.shape[0] // 2

                start_idx = mbi * self.mb_size
                end_idx = (mbi + 1) * self.mb_size
                sliced_data[key] = torch.cat(
                    [value[start_idx:end_idx], value[start_idx + bs : end_idx + bs]]
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
class AccelerateRDPOTrainer(AccelerateRLTrainer):
    """RDPO Accelerate Trainer"""

    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self.config = config

        self.lm_loss_fn = torch.nn.CrossEntropyLoss()
        self.log_lambda = torch.tensor([self.config.method.log_lambda])
        # self.moving_average_lopP = torch.tensor([self.config.method.moving_average_lopP])
        self.beta = torch.tensor([self.config.method.dpo_beta])
        # self.moving_average_lopP = torch.tensor([0.0])

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
            if self.config.method.freeze_emb:
                freeze_emb_causal_layers(model.base_model)

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
        )

    def loss(self, batch):
        """batch:  bs*chosen + bs*reject"""
        # paddding_side: left
        starts = batch['s_res'] - 1
        end = -1

        n_samples = batch['input_ids'].shape[0]
        n_queries = n_samples // 2
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
        ref_logprobs = logprobs_of_labels(
            ref_logits[:, :-1, :], batch['input_ids'][:, 1:]
        )
        N_vocab = logits.shape[-1]

        # [2 * n_queries ,] This is pi(y|x)
        logprobs_response = []
        ref_logprobs_response = []
        avg_values = []

        LM_loss_w = 0.0
        LM_loss_l = 0.0

        for ix in range(n_samples):
            logp = logprobs[ix, starts[ix] : end].mean().unsqueeze(dim=0)
            rlogp = ref_logprobs[ix, starts[ix] : end].mean().unsqueeze(dim=0)
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
        self.beta = max(torch.exp(self.beta.log() * 1.01), torch.tensor([0.1]))
        beta = self.beta.item()
        # beta = self.config.method.dpo_beta

        ## RM的计算方式
        if '[hard_reward]' in self.config.method.rdpo_type:
            rm_score = torch.ones_like(avg_values) * self.config.method.margin_rm
            rm_score[n_queries:] = -1.0 * self.config.method.margin_rm
        elif '[score_reward]' in self.config.method.rdpo_type:
            rm_score = avg_values.detach()

        elif '[score_reward_norm]' in self.config.method.rdpo_type:
            rm_score = avg_values.detach() / avg_values.detach().norm()

        elif '[delta_reward]' in self.config.method.rdpo_type:
            # 正负1初始化
            rm_score = torch.ones_like(avg_values)
            rm_score[n_queries:] = -1.0
            # 以差值调整
            delta_score = (
                avg_values.detach()[:n_queries] - avg_values.detach()[n_queries:]
            )
            for t in range(n_queries):
                if delta_score[t] > 0:
                    rm_score[t] = delta_score[t] * 0.5
                    rm_score[t + n_queries] = -delta_score[t] * 0.5

            # 以margin截断
            rm_score = torch.clamp(
                rm_score, -self.config.method.margin_rm, self.config.method.margin_rm
            )

        elif '[delta_reward_norm]' in self.config.method.rdpo_type:
            # 正负1初始化
            rm_score = torch.ones_like(avg_values)
            rm_score[n_queries:] = -1.0
            # 以差值调整(使用归一化reward计算)
            avg_values_norm = avg_values.detach() / avg_values.detach().norm()
            delta_score = avg_values_norm[:n_queries] - avg_values_norm[n_queries:]
            for t in range(n_queries):
                if delta_score[t] > 0:
                    rm_score[t] = delta_score[t] * 0.5
                    rm_score[t + n_queries] = -delta_score[t] * 0.5

            # 以margin截断
            rm_score = torch.clamp(
                rm_score, -self.config.method.margin_rm, self.config.method.margin_rm
            )

        ############ for COPF #########
        elif '[gaussian_reward]' in self.config.method.rdpo_type:
            rm_score = torch.randn(
                avg_values.shape, dtype=avg_values.dtype, device=avg_values.device
            ).view([2, -1])
            rm_score = rm_score.sort(dim=0)[0].view(-1)
            rm_score = rm_score * (2**0.5)
        elif '[linear_reward]' in self.config.method.rdpo_type:
            rm_score = torch.ones_like(avg_values) * self.config.method.margin_rm
            rm_score[n_queries:] = -1.0 * self.config.method.margin_rm
        ###############################

        else:
            NotImplementedError(
                f"rdpo_type='{self.config.method.rdpo_type}' is not implemented"
            )

        ## 初始化配分函数Z与最优解
        Z = torch.zeros([1])
        optimal_sol = torch.zeros([1])

        ## RDPO type
        if '[ref_rdpo]' in self.config.method.rdpo_type:
            Z_half = ref_logprobs_response.exp() * (1 / beta * rm_score).exp()
            Z = Z_half + torch.cat([Z_half[n_queries:], Z_half[:n_queries]], dim=0)
            optimal_sol = ref_logprobs_response + (1 / beta) * rm_score - Z.log()
            RDPO_loss = (logprobs_response - optimal_sol).square().mean()

        elif '[policy_rdpo]' in self.config.method.rdpo_type:
            Z_half = logprobs_response.exp() * (1 / beta * rm_score).exp()
            Z = Z_half + torch.cat([Z_half[n_queries:], Z_half[:n_queries]], dim=0)
            optimal_sol = logprobs_response + (1 / beta) * rm_score - Z.log()
            RDPO_loss = (logprobs_response - optimal_sol).square().mean()

        elif '[dvpo]' in self.config.method.rdpo_type:
            if self.config.method.margin_dpo > 0:
                ratio_clip = torch.clamp(
                    -self.config.method.margin_dpo
                    + logprobs_response[:n_queries]
                    - logprobs_response[n_queries:],
                    -self.config.method.cliprange,
                    0.0,
                )
            else:
                ratio_clip = (
                    logprobs_response[:n_queries] - logprobs_response[n_queries:]
                )
            RDPO_loss = (
                -beta
                * (
                    torch.sigmoid(avg_values[n_queries:] - avg_values[:n_queries])
                    * ratio_clip
                ).mean()
            )

        elif '[ref_kldpo]' in self.config.method.rdpo_type:
            Z_half = ref_logprobs_response.exp() * (1 / beta * rm_score).exp()
            Z = Z_half + torch.cat([Z_half[n_queries:], Z_half[:n_queries]], dim=0)
            optimal_sol = ref_logprobs_response + (1 / beta) * rm_score - Z.log()
            RDPO_loss = F.kl_div(
                logprobs_response.reshape(2, n_queries).t().log_softmax(dim=-1),
                optimal_sol.exp().reshape(2, n_queries).t().softmax(dim=-1),
                reduction='mean',
            )

        ############ for rank-based approach #############

        elif '[dpo]' in self.config.method.rdpo_type:
            RDPO_loss = -torch.sigmoid(beta * ratio_w - beta * ratio_l).log().mean()

        elif '[copf_kl]' in self.config.method.rdpo_type:
            scale_optimal = (1 / beta * rm_score.detach()).exp()
            P_policy_star = (
                logprobs_response.detach().exp()
                * scale_optimal
                / (logprobs_response.detach().exp() * scale_optimal).sum()
            )
            P_policy = logprobs_response.exp() / logprobs_response.exp().sum()
            RDPO_loss = F.kl_div(P_policy.log(), P_policy_star, reduction='mean')
        elif '[copf_l2]' in self.config.method.rdpo_type:
            scale_optimal = (1 / beta * rm_score.detach()).exp()
            P_policy_star = (
                logprobs_response.detach().exp()
                * scale_optimal
                / (logprobs_response.detach().exp() * scale_optimal).sum()
            )
            P_policy = logprobs_response.exp() / logprobs_response.exp().sum()
            RDPO_loss = (P_policy.log() - P_policy_star.log()).square().mean()

        elif '[PRO]' in self.config.method.rdpo_type:
            P_policy = logprobs_response.view([2, -1]).exp() / logprobs_response.view(
                [2, -1]
            ).exp().sum(dim=0)
            RDPO_loss = -P_policy.log()[0].mean()

        elif '[RRHF]' in self.config.method.rdpo_type:
            RDPO_loss = torch.clamp(
                logprobs_response[n_queries:] - logprobs_response[:n_queries],
                0.0,
                100.0,
            ).mean()

        ##################################################

        else:
            NotImplementedError(
                f"rdpo_type='{self.config.method.rdpo_type}' is not implemented"
            )

        ## regularization
        reg_loss = (logprobs_response - ref_logprobs_response).square().mean()

        # stablize loss
        # if self.moving_average_lopP == 0.0:
        #     self.moving_average_lopP = logprobs_response.sum().detach()
        # self.moving_average_lopP = 0.9 * self.moving_average_lopP + 0.1 * logprobs_response.sum().detach()
        # stb_loss = (logprobs_response.sum() - self.moving_average_lopP).square()/ (n_samples*n_samples)

        # use Lagrangian
        if self.config.method.constraint_threshold > 0:
            ## Lagrangian method for updating lambda
            Jc = (reg_loss - self.config.method.constraint_threshold).cpu().item()
            # log lambda(k+1) = log lambda(k+1) + alpha* lambda(k+1) * Jc
            self.log_lambda = torch.clamp(
                self.log_lambda
                + self.config.method.lambda_lr * self.log_lambda.exp() * Jc,
                -1.0,
                1.0,
            )

            new_lambda = self.log_lambda.exp().cpu().item()
            norm_coef = (
                new_lambda + self.config.method.coef_dpo + self.config.method.coef_nll
            )

            loss = (
                self.config.method.coef_rm * RW_loss
                + self.config.method.coef_nll / norm_coef * LM_loss_w
                + self.config.method.coef_dpo / norm_coef * RDPO_loss
                + self.config.method.coef_reg / norm_coef * reg_loss
            )  # + \
        # self.config.method.coef_stb * stb_loss

        else:  # not use Lagrangian
            loss = (
                self.config.method.coef_rm * RW_loss
                + self.config.method.coef_nll * LM_loss_w
                + self.config.method.coef_dpo * RDPO_loss
                + self.config.method.coef_reg * reg_loss
            )  # + \
            # 1.0 * stb_loss

        ## wandb
        stats = dict(
            losses=dict(
                total_loss=loss.item(),
                RDPO_loss=RDPO_loss.item(),
                # stb_loss = stb_loss.item(),
                reg_loss=reg_loss.item(),
                RW_loss=RW_loss.item(),
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
                # beta=self.beta,
                # moving_average_lopP=self.moving_average_lopP,
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
                lagrangian_multiplier=new_lambda
                if self.config.method.constraint_threshold > 0
                else 0.0,
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
        self.store = DVPOPipeline(samples, input_length, output_length, self.tokenizer)

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
        N_oom_errors = 0

        try:
            os.makedirs(self.config.train.checkpoint_dir)
        except:
            pass
        # For each epoch
        for _ in range(self.config.train.epochs):
            # For each batch
            for mbs in RDPOMiniBatchIterator(
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
                    n_oom_errors = 0  # 记录oom次数
                    batch_oom = 0

                    for mb in mbs:
                        with self._accumulate():
                            try:
                                forward_time -= time()
                                loss, stats = self.loss(mb)
                                forward_time += time()
                                backward_time -= time()
                                self.accelerator.backward(loss)
                                backward_time += time()
                                stats_accum.append(stats)
                            except RuntimeError as exception:  # 个别oom可以跳过
                                if 'out of memory' in str(exception):
                                    n_oom_errors += 1
                                    print(
                                        f'WARNING: out of memory, times:{n_oom_errors}/{self.num_mb}.'
                                    )
                                    torch.save(
                                        mb,
                                        f'{self.config.train.checkpoint_dir}/OOM-N{N_oom_errors}_n{n_oom_errors}.pt',
                                    )
                                    if n_oom_errors == self.num_mb:  # oom达到最大次数
                                        # raise exception
                                        N_oom_errors += 1
                                        batch_oom = 1
                                        print(
                                            f'WARNING: out of memory, global oom setps: {N_oom_errors}'
                                        )
                                    if hasattr(torch.cuda, 'empty_cache'):
                                        torch.cuda.empty_cache()
                                else:
                                    raise exception
                    if batch_oom == 1:
                        self.opt.zero_grad()
                        continue

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
                        else:  # 节约显存的保存方案
                            device = self.model.device  #
                            self.model.to('cpu')
                            self.save_pretrained(directory)
                            self.model.to(device)

                    stats['time/forward'] = forward_time
                    stats['time/backward'] = backward_time
                    for group_number, lr in enumerate(self.scheduler.get_last_lr()):
                        stats[f'learning_rate_group_{group_number}'] = lr

                    if (
                        self.iter_count % self.config.train.eval_interval == 0
                        or self.iter_count >= self.total_steps
                    ):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
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
                                else:  # 节约显存的保存方案
                                    device = self.model.device  #
                                    self.model.to('cpu')
                                    self.save_pretrained(directory)
                                    self.model.to(device)

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
