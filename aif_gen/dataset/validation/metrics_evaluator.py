from typing import Callable, Dict, List

import nltk
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline

from aif_gen.dataset import AlignmentDataset

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class AlignmentEvaluator:
    def __init__(self) -> None:
        self.classifier = pipeline('sentiment-analysis')

    def get_alignment_mode(self, dataset: AlignmentDataset) -> Callable[[dict], float]:
        """Looks at the dataset's task preference and returns a function that takes
        a classifier result and outputs a score between 0 and 1.

        - If the preference contains "polarizing": returns the polarizing score,
          computed as abs(score - 0.5)*2.
        - If the preference mentions "negative": favors a NEGATIVE result.
        - If the preference mentions "positive": favors a POSITIVE result.
        - Otherwise, defaults to favoring a POSITIVE result.
        """
        preference = (
            dataset.task.preference.lower()
            if hasattr(dataset.task, 'preference')
            else ''
        )
        if 'polarizing' in preference:
            return lambda result: abs(result['score'] - 0.5) * 2
        elif 'negative' in preference:
            return (
                lambda result: result['score']
                if result['label'] == 'NEGATIVE'
                else 1 - result['score']
            )
        elif 'positive' in preference:
            return (
                lambda result: result['score']
                if result['label'] == 'POSITIVE'
                else 1 - result['score']
            )
        else:
            # Default to assuming a positive response is desired.
            return (
                lambda result: result['score']
                if result['label'] == 'POSITIVE'
                else 1 - result['score']
            )

    def evaluate(self, dataset: AlignmentDataset) -> List[Dict[str, int]]:
        """For each sample, run the sentiment classifier on the chosen response.
        Use the alignment function (determined by the task's preference) to compute the score.

        Returns:
            List[Dict[str, int]]: Each dictionary has the key "alignment" with an integer percentage score.
        """
        mode_func = self.get_alignment_mode(dataset)
        results = []
        for sample in dataset.samples:
            chosen = sample.chosen
            result = self.classifier(chosen)[0]
            score = mode_func(result)
            score_int = int(round(score * 100))
            results.append({'alignment': score_int})
        return results


class ContrastEvaluator:
    """Computes the difference between the alignment scores of the chosen and rejected responses.
    It uses the same alignment function (based on the task preference) for both.
    """

    def __init__(self) -> None:
        self.classifier = pipeline('sentiment-analysis')

    def get_alignment_mode(self, dataset: AlignmentDataset) -> Callable[[dict], float]:
        """Identical to the one in AlignmentEvaluator: returns a function mapping a classifier result to a score."""
        preference = (
            dataset.task.preference.lower()
            if hasattr(dataset.task, 'preference')
            else ''
        )
        if 'polarizing' in preference:
            return lambda result: abs(result['score'] - 0.5) * 2
        elif 'negative' in preference:
            return (
                lambda result: result['score']
                if result['label'] == 'NEGATIVE'
                else 1 - result['score']
            )
        elif 'positive' in preference:
            return (
                lambda result: result['score']
                if result['label'] == 'POSITIVE'
                else 1 - result['score']
            )
        else:
            return (
                lambda result: result['score']
                if result['label'] == 'POSITIVE'
                else 1 - result['score']
            )

    def evaluate(self, dataset: AlignmentDataset) -> List[Dict[str, int]]:
        """For each sample, compute the alignment score for both the chosen and rejected responses
        (using the alignment function determined by the dataset's preference) and return the difference.

        Returns:
            List[Dict[str, int]]: Each dictionary has the key "contrast" with the computed difference as an integer percentage.
        """
        mode_func = self.get_alignment_mode(dataset)
        results = []
        for sample in dataset.samples:
            chosen = sample.chosen
            rejected = sample.rejected
            result_chosen = self.classifier(chosen)[0]
            result_rejected = self.classifier(rejected)[0]
            score_chosen = mode_func(result_chosen)
            score_rejected = mode_func(result_rejected)
            contrast = score_chosen - score_rejected
            score_int = int(round(contrast * 100))
            results.append({'contrast': score_int})
        return results


class RelevanceEvaluator:
    def __init__(self) -> None:
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def evaluate(self, dataset: AlignmentDataset) -> List[Dict[str, int]]:
        """For each sample in the AlignmentDataset, compute the cosine similarity between
        the prompt and the chosen response.

        Returns:
            List[Dict[str, int]]: A list of dictionaries with key "relevance".
        """
        results = []
        for sample in dataset.samples:
            prompt = sample.prompt
            chosen = sample.chosen
            embeddings = self.model.encode([prompt, chosen], convert_to_tensor=True)
            cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            score_int = int(round(cosine_sim * 100))
            results.append({'relevance': score_int})
        return results


class CoherenceEvaluator:
    def __init__(self) -> None:
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

    def evaluate(self, dataset: AlignmentDataset) -> List[Dict[str, int]]:
        """For each sample, compute the coherence score (inverse perplexity) of the chosen response.

        Returns:
            List[Dict[str, int]]: A list of dictionaries with key "coherence".
        """
        results = []
        for sample in dataset.samples:
            chosen = sample.chosen
            encodings = self.tokenizer(chosen, return_tensors='pt')
            input_ids = encodings.input_ids
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
            perplexity = torch.exp(loss).item()
            coherence_score = 1.0 / perplexity if perplexity > 0 else 0.0
            score_int = int(round(coherence_score * 100))
            results.append({'coherence': score_int})
        return results


def extract_noun_chunks_nltk(text: str) -> set:
    """Uses NLTK's POS tagger and a simple chunk grammar to extract noun phrases.

    Returns:
        A set of noun chunks (lowercased and stripped) found in the text.
    """
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)

    # Define a simple chunk grammar for noun phrases (NP)
    grammar = 'NP: {<DT>?<JJ>*<NN.*>+}'
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(tagged_tokens)

    noun_chunks = set()
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        chunk = ' '.join(word for word, pos in subtree.leaves())
        noun_chunks.add(chunk.lower().strip())
    return noun_chunks


class CoverageEvaluator:
    def evaluate(self, dataset: AlignmentDataset) -> List[Dict[str, int]]:
        """For each sample in the dataset, extract noun chunks from the prompt using NLTK
        and compute the fraction that appear in the chosen response.

        Returns:
            List[Dict[str, int]]: A list of dictionaries with key "coverage".
        """
        results = []
        for sample in dataset.samples:
            prompt = sample.prompt
            chosen = sample.chosen.lower()
            # Use NLTK to extract noun chunks
            keywords = extract_noun_chunks_nltk(prompt)
            if not keywords:
                ratio = 1.0
            else:
                # Count how many noun chunks appear in the chosen response.
                covered = sum(1 for kw in keywords if kw in chosen)
                ratio = covered / len(keywords)
            score_int = int(round(ratio * 100))
            results.append({'coverage': score_int})
        return results


# def calc_bleu(response, hypothesis, weight) -> float:
#     """Computes the BLEU score for a single response against others.
#     Needs to be declared outside of class for parallel processing.
#     """
#     return sentence_bleu(
#         response, hypothesis, weight, smoothing_function=SmoothingFunction().method1
#     )


# class DiversityEvaluator:
#     def compute_response_diversity(
#         self, response_set: list[str], ngram: int = 3, parallel: bool = False
#     ) -> float:
#         """Computes the diversity for a set of responses via the inverse Self-BLEU score.
#         A higher Self-BLEU score indicates lower diversity for generated sentences,
#         therefore we return the inverse score.
#         https://arxiv.org/pdf/1802.01886
#         https://github.com/geek-ai/Texygen

#         Args:
#             response_set (list[str]): A list of generated sentences.
#             ngram (int): The maximum n-gram order for BLEU calculation. Default of 3 matches the original paper.
#             parallel (bool): Whether to compute BLEU scores in using Pool multiprocessing.
#         """
#         # Avoid redundant calculations
#         if not response_set or len(response_set) < 2:
#             return 0.0

#         # BLEU weight setting (e.g., for BLEU-3: (1/3, 1/3, 1/3))
#         weight = tuple(1.0 / ngram for _ in range(ngram))

#         # Tokenize responses for BLEU calculation
#         tokenized_responses = [
#             nltk.word_tokenize(sentence) for sentence in response_set
#         ]

#         scores = []

#         if parallel:
#             # Compute BLEU scores in parallel
#             with Pool() as pool:
#                 scores = pool.starmap(
#                     calc_bleu,
#                     [
#                         (
#                             tokenized_responses[:i] + tokenized_responses[i + 1 :],
#                             hypothesis,
#                             weight,
#                         )
#                         for i, hypothesis in enumerate(tokenized_responses)
#                     ],
#                 )
#         else:
#             # Compute BLEU scores sequentially
#             for i, hypothesis in enumerate(tokenized_responses):
#                 other_responses = tokenized_responses[:i] + tokenized_responses[i + 1 :]
#                 score = calc_bleu(other_responses, hypothesis, weight)
#                 scores.append(score)

#         # Average self-BLEU score
#         bleu_score = sum(scores) / len(scores)

#         # Return the inverse BLEU score as diversity metric
#         return 1.0 - bleu_score
