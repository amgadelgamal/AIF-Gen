import torch
import torch.nn.functional as F
import transformers
import trlx.utils.logging as logging

from ...trlx.data.configs import TRLConfig
from ...trlx.models.modeling_dvpo import (
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
)
from ...trlx.pipeline.dvpo_pipeline import DVPOPipeline
from ...trlx.trainer import register_trainer
from ...trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from ...trlx.utils.modeling import logprobs_of_labels

logger = logging.get_logger(__name__)


def logprobs_of_labels_fp32(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits.
    """
    logprobs = F.log_softmax(logits.float(), dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).half()
    return logprobs_labels.squeeze(-1)


@register_trainer
class AccelerateDVPOTrainer(AccelerateRLTrainer):
    """DPO Accelerate Trainer"""

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
        # policy model forward()
        outputs = self.model(
            batch['input_ids'], batch['attention_mask'], return_dict=True
        )
        logits = outputs.logits
        # [B,L]
        value = outputs.value

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

        # paddding_side: left
        starts = batch['s_res'] - 1
        end = -1

        n_samples = batch['input_ids'].shape[0]
        n_queries = n_samples // 2
        assert (
            n_samples == n_queries * 2
        ), f'not equal: n_samples={n_samples},n_queries={n_queries}'

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

        # [n_queries, ] log (pi(y|x)/pi_ref(y|x)) = log pi(y|x) - log pi_ref(y|x)
        ratio_w = logprobs_response[:n_queries] - ref_logprobs_response[:n_queries]
        ratio_l = logprobs_response[n_queries:] - ref_logprobs_response[n_queries:]

        # rm loss
        rm_clip = torch.clamp(
            -self.config.method.margin_rm
            + avg_values[:n_queries]
            - avg_values[n_queries:],
            -self.config.method.cliprange_reward,
            0.0,
        )
        RW_loss = -torch.log(torch.sigmoid(rm_clip)).mean()

        rm_acc = sum(avg_values[:n_queries] > avg_values[n_queries:]) / (
            avg_values.shape[0] / 2
        )

        # hyper-parameter
        beta = self.config.method.dpo_beta
        ratio_clip = torch.clamp(
            -self.config.method.margin_dpo
            + logprobs_response[:n_queries]
            - logprobs_response[n_queries:],
            -self.config.method.cliprange,
            0.0,
        )

        DPO_loss = (
            -beta
            * (
                torch.sigmoid(avg_values[n_queries:] - avg_values[:n_queries])
                * (ratio_clip)
            ).mean()
        )

        loss = (
            self.config.method.coef_rm * RW_loss
            + self.config.method.coef_nll * LM_loss_w
            + self.config.method.coef_dpo * DPO_loss
        )

        stats = dict(
            loss=loss.item(),
            DPO_loss=DPO_loss.item(),
            RW_loss=RW_loss.item(),
            rm_acc=rm_acc.item(),
            LM_loss_w=LM_loss_w.item(),
            LM_loss_l=LM_loss_l.item(),
            ratio_w=ratio_w.mean().item(),
            ratio_l=ratio_l.mean().item(),
            socre_w=avg_values[:n_queries].mean().item(),
            socre_l=avg_values[n_queries:].mean().item(),
            logprobs_w=logprobs_response[:n_queries].mean().item(),
            ref_logprobs_w=ref_logprobs_response[:n_queries].mean().item(),
            logprobs_l=logprobs_response[n_queries:].mean().item(),
            ref_logprobs_l=ref_logprobs_response[n_queries:].mean().item(),
        )
        return loss, stats

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
