"""
ConvoKit corpus loader for argument evaluation.

Loads argument pairs from ConvoKit corpora like the Winning Arguments corpus
(ChangeMyView) and converts them to our Argument format for evaluation.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from convokit import Corpus, download

from .models import Argument


@dataclass
class ArgumentPair:
    """A pair of arguments: one successful, one unsuccessful."""
    successful: Argument
    unsuccessful: Argument
    pair_id: str
    conversation_id: str
    topic: str  # The original post title/topic


@dataclass
class CMVConversation:
    """A ChangeMyView conversation with its argument pairs."""
    id: str
    title: str
    op_text: str
    op_user: str
    pairs: list[ArgumentPair] = field(default_factory=list)


class WinningArgumentsLoader:
    """
    Loader for the Winning Arguments (ChangeMyView) corpus.

    This corpus contains paired arguments from Reddit's r/ChangeMyView:
    - Successful arguments that changed the OP's mind (awarded delta)
    - Unsuccessful arguments that didn't change the OP's mind

    Perfect for testing if AI judges can distinguish persuasive arguments.
    """

    CORPUS_NAME = "winning-args-corpus"

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize the loader.

        Args:
            data_dir: Directory to store downloaded corpus. If None, uses ConvoKit default.
        """
        self.data_dir = data_dir
        self._corpus: Corpus | None = None

    @property
    def corpus(self) -> Corpus:
        """Lazy-load the corpus."""
        if self._corpus is None:
            print(f"Loading {self.CORPUS_NAME}...")
            if self.data_dir:
                corpus_path = self.data_dir / self.CORPUS_NAME
                if corpus_path.exists():
                    self._corpus = Corpus(filename=str(corpus_path))
                else:
                    self._corpus = Corpus(filename=download(self.CORPUS_NAME, data_dir=str(self.data_dir)))
            else:
                self._corpus = Corpus(filename=download(self.CORPUS_NAME))
            print(f"  Loaded {len(self._corpus.get_utterance_ids())} utterances")
            print(f"  Loaded {len(self._corpus.get_conversation_ids())} conversations")
        return self._corpus

    def get_stats(self) -> dict:
        """Get corpus statistics."""
        corpus = self.corpus

        # Count successful/unsuccessful utterances
        successful = 0
        unsuccessful = 0
        for utt in corpus.iter_utterances():
            success = utt.meta.get("success")
            if success == 1:
                successful += 1
            elif success == 0:
                unsuccessful += 1

        return {
            "utterances": len(corpus.get_utterance_ids()),
            "conversations": len(corpus.get_conversation_ids()),
            "speakers": len(corpus.get_speaker_ids()),
            "successful_arguments": successful,
            "unsuccessful_arguments": unsuccessful,
        }

    def _create_argument_id(self, utterance_id: str, success: bool) -> str:
        """Create a stable argument ID from utterance."""
        prefix = "cmv_win_" if success else "cmv_lose_"
        # Use first 8 chars of hash for shorter IDs
        hash_val = hashlib.md5(utterance_id.encode()).hexdigest()[:8]
        return f"{prefix}{hash_val}"

    def _utterance_to_argument(
        self,
        utterance,
        topic: str,
        success: bool,
    ) -> Argument | None:
        """Convert a ConvoKit utterance to our Argument format."""
        text = utterance.text
        if not text or text.strip() == "" or utterance.speaker.id == "[missing]":
            return None

        # For CMV, we don't have explicit stances - arguments are responses to OP
        # We mark successful as "for" (supporting change) and unsuccessful as "against"
        stance: Literal["for", "against"] = "for" if success else "against"

        # For CMV arguments, we don't have injected faults or expected scores
        # The "ground truth" is whether it won a delta or not
        return Argument(
            id=self._create_argument_id(utterance.id, success),
            topic=topic,
            stance=stance,
            text=text,
            injected_faults=[],  # No known faults
            expected_score=100 if success else 50,  # Successful = high score expected
            source="curated",
            generated_by=None,
        )

    def iter_pairs(
        self,
        max_pairs: int | None = None,
        min_text_length: int = 100,
        max_text_length: int = 2000,
    ) -> Iterator[ArgumentPair]:
        """
        Iterate over argument pairs (successful vs unsuccessful).

        Args:
            max_pairs: Maximum number of pairs to return
            min_text_length: Minimum text length to include
            max_text_length: Maximum text length to include

        Yields:
            ArgumentPair objects
        """
        corpus = self.corpus
        count = 0

        for conv in corpus.iter_conversations():
            conv_id = conv.id
            title = conv.meta.get("op-title", "Unknown Topic")
            pair_ids = conv.meta.get("pair_ids", [])

            # Get all utterances in this conversation indexed by ID
            utt_lookup = {utt.id: utt for utt in conv.iter_utterances()}

            # Process each pair
            for pair_id in pair_ids:
                # Find successful and unsuccessful utterances for this pair
                successful_utt = None
                unsuccessful_utt = None

                for utt in conv.iter_utterances():
                    utt_pair_ids = utt.meta.get("pair_ids", [])
                    if pair_id in utt_pair_ids:
                        success = utt.meta.get("success")
                        if success == 1:
                            successful_utt = utt
                        elif success == 0:
                            unsuccessful_utt = utt

                # Create pair if we have both
                if successful_utt and unsuccessful_utt:
                    # Apply length filters
                    succ_text = successful_utt.text or ""
                    unsucc_text = unsuccessful_utt.text or ""

                    if not (min_text_length <= len(succ_text) <= max_text_length):
                        continue
                    if not (min_text_length <= len(unsucc_text) <= max_text_length):
                        continue

                    succ_arg = self._utterance_to_argument(successful_utt, title, True)
                    unsucc_arg = self._utterance_to_argument(unsuccessful_utt, title, False)

                    if succ_arg and unsucc_arg:
                        yield ArgumentPair(
                            successful=succ_arg,
                            unsuccessful=unsucc_arg,
                            pair_id=pair_id,
                            conversation_id=conv_id,
                            topic=title,
                        )

                        count += 1
                        if max_pairs and count >= max_pairs:
                            return

    def get_pairs(
        self,
        max_pairs: int | None = None,
        min_text_length: int = 100,
        max_text_length: int = 2000,
    ) -> list[ArgumentPair]:
        """Get a list of argument pairs."""
        return list(self.iter_pairs(max_pairs, min_text_length, max_text_length))

    def get_arguments(
        self,
        max_pairs: int | None = None,
        min_text_length: int = 100,
        max_text_length: int = 2000,
    ) -> list[Argument]:
        """
        Get flat list of all arguments (both successful and unsuccessful).

        Args:
            max_pairs: Maximum number of pairs (returns 2x this many arguments)
            min_text_length: Minimum text length
            max_text_length: Maximum text length

        Returns:
            List of Argument objects
        """
        arguments = []
        for pair in self.iter_pairs(max_pairs, min_text_length, max_text_length):
            arguments.append(pair.successful)
            arguments.append(pair.unsuccessful)
        return arguments

    def sample_pairs(
        self,
        n: int = 10,
        seed: int | None = None,
        min_text_length: int = 100,
        max_text_length: int = 2000,
    ) -> list[ArgumentPair]:
        """
        Get a random sample of argument pairs.

        Args:
            n: Number of pairs to sample
            seed: Random seed for reproducibility
            min_text_length: Minimum text length
            max_text_length: Maximum text length

        Returns:
            List of sampled ArgumentPair objects
        """
        import random

        if seed is not None:
            random.seed(seed)

        # Get all pairs first (with reasonable limit for memory)
        all_pairs = self.get_pairs(
            max_pairs=1000,
            min_text_length=min_text_length,
            max_text_length=max_text_length,
        )

        if len(all_pairs) <= n:
            return all_pairs

        return random.sample(all_pairs, n)


def list_available_corpora() -> list[dict]:
    """List ConvoKit corpora that may be useful for argument evaluation."""
    return [
        {
            "name": "winning-args-corpus",
            "description": "ChangeMyView successful vs unsuccessful argument pairs",
            "size": "34,911 speakers, 293,297 utterances, 3,051 conversations",
            "loader": "WinningArgumentsLoader",
        },
        {
            "name": "conversations-gone-awry-cmv-corpus",
            "description": "CMV conversations that derailed vs stayed on track",
            "size": "Subset of CMV with derailment labels",
            "loader": None,  # Not implemented yet
        },
        {
            "name": "persuasion-reddit-corpus",
            "description": "Reddit persuasion conversations",
            "size": "Various persuasion attempts",
            "loader": None,  # Not implemented yet
        },
    ]
