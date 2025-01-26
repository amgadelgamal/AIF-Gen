import json
import os
import uuid
from time import time
from typing import Callable, List

import ray
import torch
import torch.nn.functional as F
import transformers
import trlx.utils.logging as logging
from ray.air import session
from torch.utils.data import DataLoader

from benchmarks.COPR.trlx.data.accelerate_base_datatypes import PromptBatch
from benchmarks.COPR.trlx.data.configs import TRLConfig

# from benchmarks.COPR.trlx.data.ppo_types import PPORLBatch, PPORLElement
from benchmarks.COPR.trlx.data.spin_types import SPINBatch, SPINElement
from benchmarks.COPR.trlx.models.modeling_spin import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    FixedKLController,
)
from benchmarks.COPR.trlx.pipeline import MiniBatchIterator

# from benchmarks.COPR.trlx.pipeline.ppo_pipeline import PPORolloutStorage
from benchmarks.COPR.trlx.pipeline.spin_pipeline import (
    SPINPipeline,
    SPINRolloutStorage,
)
from benchmarks.COPR.trlx.trainer import register_trainer
from benchmarks.COPR.trlx.trainer.accelerate_base_trainer import (
    AccelerateRLTrainer,
)
from benchmarks.COPR.trlx.utils import filter_non_scalars, infinite_dataloader
from benchmarks.COPR.trlx.utils.modeling import (
    RunningMoments,
    logprobs_of_labels,
)

logger = logging.get_logger(__name__)


@register_trainer
class AccelerateSPINTrainer(AccelerateRLTrainer):
    """SPIN Accelerate Trainer
    SPIN: 对抗版本的DPO
    """

    reward_fn: Callable[[List[str], List[str], List[str]], List[float]]
    # tokenizer: AutoTokenizer

    def __init__(self, config: TRLConfig, **kwargs):
        """Accelerate Trainer initialization
        Args:
            config: Config
        """
        super().__init__(config, **kwargs)

        self.lm_loss_fn = torch.nn.CrossEntropyLoss()

        # Setup rollout logging
        if config.train.rollout_logging_dir is not None:
            self.log_rollouts = True
            self.setup_rollout_logging(config)
        else:
            self.log_rollouts = False

        # Setup the rollout store
        # Rollouts contain the prompt & response, log probs, values and rewards - from each rollout
        self.store = SPINRolloutStorage(
            self.tokenizer.pad_token_id, self.tokenizer.padding_side
        )

        # Create the rollout store dataloader (for batching up rollouts)
        # TODO (jon-tow): This is only used to satisfy to `accelerator.prepare` call constraint below - remove in future
        rollout_loader: DataLoader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        # Prepare multi-GPU acceleration
        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()  # Clear the rollout store

        # Setup a reference model when hydra heads are not used
        # if not hasattr(self.model, "frozen_head"):
        #     self.ref_model = self.get_arch(self.config)
        #     self.ref_model.to(self.accelerator.device)
        #     self.ref_model.eval()

        # Setup the KL controller
        # This helps prevent large divergences in the controller (policy)
        if config.method.target is not None:
            self.kl_ctl = AdaptiveKLController(
                config.method.init_kl_coef, config.method.target, config.method.horizon
            )
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        # Create the parameters for the Hugging Face language model's generator
        # method (that generates new tokens from a prompt).
        # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin.generate

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if config.method.gen_experience_kwargs is not None:
            self.generate_experience_kwargs = dict(
                config.method.gen_experience_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            self.generate_experience_kwargs = None

        # Setup stats tracker
        self.running_moments = RunningMoments()
        self.ref_mean = self.config.method.ref_mean
        self.ref_std = self.config.method.ref_std

    def get_arch(self, config: TRLConfig):
        """Get the model"""
        model_class = AutoModelForCausalLMWithHydraValueHead

        from_fn = model_class.from_pretrained
        # backward-compat: Try to create a randomly initialized architecture from a config
        if issubclass(type(config.model.model_path), transformers.PretrainedConfig):
            from_fn = model_class.from_config

        return from_fn(
            config.model.model_path,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
        )

    def learn(self):  # noqa: C901
        """Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`"""
        logger.info('Starting training')

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
            for mbs in MiniBatchIterator(
                self.train_dataloader, self.mb_size, self.num_mb
            ):
                # For each update per batch
                for _ in range(self.n_updates_per_batch):
                    # Note that whereas standard policy gradient methods perform one
                    # gradient update per batch, PPO for example commonly performs
                    # multiple gradient updates on the same batch of data.
                    # https://arxiv.org/pdf/1707.06347.pdf

                    ############原始代码##################
                    # forward_time = 0
                    # backward_time = 0
                    # stats_accum = []
                    # for mb in mbs:# mb中 奇数为正，偶数为负
                    #     with self._accumulate():
                    #         forward_time -= time()
                    #         loss, stats = self.loss(mb)
                    #         forward_time += time()
                    #         backward_time -= time()
                    #         self.accelerator.backward(loss)
                    #         backward_time += time()
                    #         stats_accum.append(stats)
                    ############原始代码##################

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
                                        f'WARNING: out of memory, times:{n_oom_errors}/{self.num_mb}. Max len of mini-batch: {mb.query_tensors.shape[1]}'
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

    def loss(self, batch: SPINBatch):
        """Forward pass & loss

        Args:
            batch: Previous batch of episodes
        """
        # Move `batch` data to `accelerator` device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        # old_logprobs = batch.logprobs.to(self.accelerator.device)
        # old_values = batch.values.to(self.accelerator.device)
        old_logprobs_response = batch.old_logp_response.to(self.accelerator.device)

        response_length = response_tensors.shape[1]

        """
        if "labels" in batch:
            labels = batch.labels.clone()
        else:
            labels = batch.input_ids.clone()
        labels[~batch.attention_mask.bool()] = -100

        loss = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=labels).loss
        """

        tokens = torch.cat((query_tensors, response_tensors), dim=1)
        attention_mask = (
            tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
        )
        outputs = self.model(tokens, attention_mask, return_dict=True)

        start = query_tensors.shape[1] - 1
        end = start + response_length

        n_samples = tokens.shape[0]
        idx1 = torch.arange(0, n_samples, 2)  # 正样本

        logits = outputs.logits
        # values_pred = outputs.value

        logprobs = logprobs_of_labels(logits[:, :-1, :], tokens[:, 1:])

        #################
        # for nll loss
        labels = tokens.clone()
        labels[~attention_mask.bool()] = -100
        shift_logits = logits[idx1, start:-1, :]
        shift_labels = labels[idx1, start + 1 :]
        nll_loss = self.lm_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        #################

        logprobs, mask = (logprobs[:, start:end], attention_mask[:, start:end])

        loss, stats = self.config.method.loss(
            logprobs=logprobs,
            # values=values_pred,
            old_logprobs_response=old_logprobs_response,
            # old_values=old_values,
            mask=mask,
            nll_loss=nll_loss,
        )

        return loss, stats

    def setup_rollout_logging(self, config):
        # Make rollout logging dir for this run and store config
        exists = os.path.exists(config.train.rollout_logging_dir)
        isdir = os.path.isdir(config.train.rollout_logging_dir)
        assert exists and isdir

        self.run_id = f'run-{uuid.uuid4()}'
        self.rollout_logging_dir = os.path.join(
            config.train.rollout_logging_dir, self.run_id
        )
        os.mkdir(self.rollout_logging_dir)

        with open(os.path.join(self.rollout_logging_dir, 'config.json'), 'w') as f:
            f.write(json.dumps(config.to_dict(), indent=2))

    def post_epoch_callback(self):
        """Post epoch callback

        Clears the store and creates `num_rollouts` new episodes.
        """
        if self.log_rollouts:
            self.store.export_history(location=self.rollout_logging_dir)
        self.store.clear_history()
        # Collect more rollouts for training
        self.make_experience(self.config.method.num_rollouts, self.iter_count)

    def post_backward_callback(self):
        # self.kl_ctl.update(self.mean_kl, n_steps=self.config.train.batch_size)
        pass

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)
        self.eval_dataloader = self.accelerator.prepare_data_loader(eval_dataloader)

        # 不能shuffle 确保正负样本交替
        self.train_dataloader = self.store.create_loader(
            self.config.train.batch_size, shuffle=False
        )

        self.n_updates_per_batch = self.config.method.ppo_epochs
        self.total_steps = (
            self.config.train.epochs
            * self.n_updates_per_batch
            * len(self.train_dataloader)
        )
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def add_prompt_pipeline(self, pipeline: SPINPipeline):
        """Add a prompt pipeline dataloader to a trainer instance for the `make_experience` stage"""
        prompt_dataloader = pipeline.create_loader(
            self.config.method.chunk_size, shuffle=True
        )
        prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
        self.prompt_iterator = infinite_dataloader(prompt_dataloader)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
        logger.info('Collecting rollouts')
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get('RANK', 0) != '0',
            desc=f'[rollout 0 / {num_rollouts}]',
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

        ppo_rl_elements = []
        accumulated_stats = []

        while len(ppo_rl_elements) < num_rollouts * 2:
            stats = {}
            # Get next batch in prompt dataset
            batch: PromptBatch = next(self.prompt_iterator)

            rollout_generate_time = time()

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            samples = self.generate(batch['input_ids'], batch['attention_mask'])
            stats['time/rollout_generate'] = time() - rollout_generate_time

            prompt_tensors = batch.input_ids
            device = samples.device

            str_samples, str_prompts, str_outputs = self.decode(
                prompt_tensors, samples, append_eos_token=True
            )

            # Pad the sample outputs
            outputs = self.tokenizer(str_outputs).input_ids
            outputs = list(map(torch.LongTensor, outputs))

            # for SPIN chosen(gold) + rejected
            double_outputs = list(map(torch.LongTensor, batch['label_ids'])) + outputs
            double_prompt_tensors = torch.cat([prompt_tensors, prompt_tensors], dim=0)

            maxsize = max(map(len, double_outputs))
            double_outputs = [
                F.pad(
                    output,
                    (
                        0,
                        maxsize - len(output),
                    ),  # 左侧补0个[PAD], 右侧补maxsize-len(output)个[PAD]
                    value=self.tokenizer.pad_token_id,
                )
                for output in double_outputs
            ]
            # double
            sample_outputs = torch.vstack(double_outputs).to(device)

            # double
            all_tokens = torch.cat(
                (double_prompt_tensors.to(device), sample_outputs), dim=1
            )

            attention_mask = (
                all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
            )
            with torch.no_grad():
                logits, *_, values = self.model(
                    all_tokens,
                    attention_mask=attention_mask,
                )
                # TODO(dahoas): When hydra model works need to also support generation on hydra head
                # if hasattr(self.model, "frozen_head"):
                #     ref_logits = self.model.forward_hydra(
                #         all_tokens,
                #         attention_mask=attention_mask,
                #         return_dict=True,
                #     ).logits
                # else:
                #     ref_logits = self.ref_model(
                #         all_tokens,
                #         attention_mask=attention_mask,
                #         return_dict=True,
                #     ).logits
                #     ref_logits = ref_logits.to(device)

            logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
            # ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

            n_samples: int = samples.shape[0]
            # Estimate the KL divergence between the model and reference model
            start = double_prompt_tensors.shape[1] - 1
            # log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]

            # ref_logprobs = ref_logprobs.cpu()
            double_prompt_tensors = double_prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            values = values.cpu()[:, :-1]

            # Get the logprobs and values, for tokens that are not padding,
            # from the start of the prompt up to the <eos> token, while also including the latter
            # (these are taken from the student model and not the reference model)
            ends = start + attention_mask[:, start:].sum(1) + 1
            all_values = [values[ix, start : ends[ix]] for ix in range(n_samples * 2)]
            all_logprobs = [
                logprobs[ix, start : ends[ix]] for ix in range(n_samples * 2)
            ]
            all_logp_responses = [
                logprobs[ix, start : ends[ix]].mean() for ix in range(n_samples * 2)
            ]

            rollout_count = 0

            # double # 奇数正样本，偶数负样本
            for sample_idx in range(n_samples):
                ppo_rl_elements.append(
                    SPINElement(
                        query_tensor=double_prompt_tensors[sample_idx],
                        response_tensor=sample_outputs[sample_idx],
                        logprobs=all_logprobs[sample_idx],
                        values=all_values[sample_idx],
                        old_logp_response=all_logp_responses[
                            sample_idx
                        ],  # SPIN 不需要reward，将这个字段用于logp_responses
                    )
                )
                ppo_rl_elements.append(
                    SPINElement(
                        query_tensor=double_prompt_tensors[sample_idx + n_samples],
                        response_tensor=sample_outputs[sample_idx + n_samples],
                        logprobs=all_logprobs[sample_idx + n_samples],
                        values=all_values[sample_idx + n_samples],
                        old_logp_response=all_logp_responses[
                            sample_idx + n_samples
                        ],  # SPIN 不需要reward，将这个字段用于logp_responses
                    )
                )

                rollout_count += 1
            accumulated_stats.append(stats)

            tbar.set_description(
                f'[rollout {len(ppo_rl_elements) // 2} / {num_rollouts}]'
            )
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        self.accelerator.log(stats, step=iter_count)
        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)
