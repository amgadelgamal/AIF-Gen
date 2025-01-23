from dataclasses import dataclass


@dataclass
class AlignmentDatasetSample:
    r"""Container for a single Alignment Dataset Sample.

    Args:
        prompt (str): The prompt associated with the sample.
        winning_response (str): The winning response associated with the sample.
        losing_response (str): The losing response associated with the sample.
    """

    prompt: str
    winning_response: str
    losing_response: str
