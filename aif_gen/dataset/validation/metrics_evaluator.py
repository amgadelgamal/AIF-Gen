import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, pipeline


class MetricsEvaluator:
    def __init__(self) -> None:
        # Load the SentenceTransformer for semantic similarity.
        self.relevance_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        # Load spaCy for keyword (noun chunk) extraction.
        self.nlp = spacy.load('en_core_web_sm')
        # Load GPT-2 for computing perplexity (as a proxy for coherence).
        self.coherence_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.coherence_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.coherence_model.eval()  # set in evaluation mode
        # Load a classifier (here using sentiment analysis as a dummy proxy for alignment).
        self.alignment_classifier = pipeline('sentiment-analysis')

    def compute_prompt_relevance_score(self, prompt: str, response: str) -> float:
        """Computes the semantic similarity between the prompt and the response.
        Higher cosine similarity means that the response is more on-topic.
        """
        embeddings = self.relevance_model.encode(
            [prompt, response], convert_to_tensor=True
        )
        cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cosine_sim.item()

    def compute_content_coherence_index(self, response: str) -> float:
        """Computes a content coherence index by calculating the GPT-2 perplexity of the response.
        Since lower perplexity indicates better coherence, we return the inverse perplexity.
        """
        encodings = self.coherence_tokenizer(response, return_tensors='pt')
        input_ids = encodings.input_ids
        with torch.no_grad():
            outputs = self.coherence_model(input_ids, labels=input_ids)
            loss = outputs.loss  # average negative log likelihood
        perplexity = torch.exp(loss).item()
        # Define the coherence index as the inverse of perplexity (avoid division by zero)
        return 1.0 / perplexity if perplexity > 0 else 0.0

    def compute_prompt_coverage_ratio(self, prompt: str, response: str) -> float:
        """Extracts key elements from the prompt (using noun chunks via spaCy) and returns
        the fraction of those keywords that appear in the response.
        """
        doc = self.nlp(prompt)
        # Extract and normalize noun chunks from the prompt.
        keywords = {
            chunk.text.lower().strip()
            for chunk in doc.noun_chunks
            if chunk.text.strip()
        }
        if not keywords:
            return 1.0  # If no keywords can be extracted, assume full coverage.
        response_lower = response.lower()
        covered = sum(1 for kw in keywords if kw in response_lower)
        return covered / len(keywords)

    def compute_response_alignment_score(self, response: str) -> float:
        """Uses a classifier to predict alignment with desired preferences.
        Here we use a sentiment analysis pipeline as a proxy:
          - If the model returns a 'POSITIVE' label, we take the score directly.
          - Otherwise, we invert the score.
        The score is returned on a 0–1 scale.
        """
        result = self.alignment_classifier(response)[0]
        if result['label'] == 'POSITIVE':
            return result['score']
        else:
            return 1 - result['score']

    def compute_response_contrast_ratio(
        self, chosen_response: str, rejected_response: str
    ) -> float:
        """Computes the difference in alignment scores between the chosen and rejected responses.
        A higher difference indicates a clearer distinction in how preferences are reflected.
        """
        score_chosen = self.compute_response_alignment_score(chosen_response)
        score_rejected = self.compute_response_alignment_score(rejected_response)
        return score_chosen - score_rejected
