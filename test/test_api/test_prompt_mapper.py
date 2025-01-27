import unittest
from aif_gen.dataset.continual_alignment_dataset import ContinualAlignmentDataset
from aif_gen.dataset.alignment_dataset import AlignmentDataset
from aif_gen.dataset.alignment_sample import AlignmentDatasetSample
from aif_gen.task import AlignmentTask
from aif_gen.task.domain import Domain
from aif_gen.task.domain_component import DomainComponent
from aif_gen.api.prompt_mapper import PromptMapper

class TestPromptMapper(unittest.TestCase):
    def setUp(self):
        """
        Set up a mock ContinualAlignmentDataset with one AlignmentDataset.
        """
        # Define domain components
        science = DomainComponent(
            name="Science",
            seed_words=["physics", "biology", "chemistry"],
            description="Scientific concepts"
        )
        communication = DomainComponent(
            name="Communication",
            seed_words=["email", "writing", "speaking"],
            description="Communication skills"
        )

        # Define domain
        domain = Domain([science, communication], [0.7, 0.3])

        # Define alignment task
        self.task = AlignmentTask(
            domain=domain,
            objective="Explain scientific concepts clearly.",
            preference="Clarity over technical detail."
        )

        # Define alignment samples
        self.samples = [
            AlignmentDatasetSample(
                prompt="Explain photosynthesis in simple terms.",
                chosen="Plants use sunlight to make food.",
                rejected="Photosynthesis is the biochemical process in plants that converts light energy to chemical energy."
            ),
            AlignmentDatasetSample(
                prompt="What is the speed of light?",
                chosen="The speed of light is approximately 300,000 km/s.",
                rejected="The speed of light is 299,792,458 m/s."
            )
        ]

        # Create alignment dataset
        self.alignment_dataset = AlignmentDataset(self.task, self.samples)

        # Create continual alignment dataset
        self.continual_dataset = ContinualAlignmentDataset([self.alignment_dataset])

    def test_generate_prompts(self):
        """
        Test that the PromptMapper generates the correct prompts.
        """
        prompt_mapper = PromptMapper()
        prompts = prompt_mapper.generate_prompts(self.continual_dataset)

        expected_prompts = [
            (
                "Objective: Explain scientific concepts clearly.\n"
                "Prompt: Explain photosynthesis in simple terms.\n"
                "Option 1: Plants use sunlight to make food.\n"
                "Option 2: Photosynthesis is the biochemical process in plants that converts light energy to chemical energy.\n"
                "Which option better aligns with the objective and why?"
            ),
            (
                "Objective: Explain scientific concepts clearly.\n"
                "Prompt: What is the speed of light?\n"
                "Option 1: The speed of light is approximately 300,000 km/s.\n"
                "Option 2: The speed of light is 299,792,458 m/s.\n"
                "Which option better aligns with the objective and why?"
            )
        ]

        # Assert the generated prompts match the expected prompts
        self.assertEqual(prompts, expected_prompts)

    def test_empty_continual_dataset(self):
        """
        Test that the PromptMapper returns an empty list when given an empty dataset.
        """
        empty_dataset = ContinualAlignmentDataset([])
        prompt_mapper = PromptMapper()
        prompts = prompt_mapper.generate_prompts(empty_dataset)

        # Assert that the output is an empty list
        self.assertEqual(prompts, [])


if __name__ == "__main__":
    unittest.main()
