# Implementation of the state-of-the-art alignment methods based on trlx

**Reproducing the codes of DPO\[1\], PRO\[2\], RRHF\[3\], SPIN (online method) \[4\], CPPO (online method) \[5\], COPF\[6\].**

\[1\] DPO: Direct Preference Optimization: Your Language Model is Secretly a Reward Model

\[2\] PRO: Preference Ranking Optimization for Human Alignment

\[3\] RRHF: Rank Responses to Align Language Models with Human Feedback without tears

\[4\] SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models

\[5\] CPPO: Continual Learning for Reinforcement Learning with Human Feedback

\[6\] COPF: CONTINUAL LEARNING HUMAN PREFERENCE THROUGH OPTIMAL POLICY FITTING

______________________________________________________________________

### Directory based on the Source code provided for COPR paper

## To run the code, you need to add the following configs to your `.env` file:

```bash
export PYTHONPATH="./benchmarks/COPR/trlx:$PYTHONPATH"
export HF_DATASETS_CACHE="$SCRATCH/.cache"
export HUGGING_FACE_TOKEN=""
export HF_HOME="$SCRATCH/.cache/huggingface"
```
