"""Data models for arguments, evaluations, and experiments."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import json
import uuid


@dataclass
class Argument:
    """A debate argument with ground truth metadata."""
    id: str
    topic: str
    stance: Literal["for", "against"]
    text: str

    # Ground truth
    injected_faults: list[str] = field(default_factory=list)
    expected_score: int = 100  # 100 - sum of fault severities

    # Metadata
    source: Literal["generated", "curated", "user"] = "generated"
    generated_by: str | None = None  # Model used for generation
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def create(
        cls,
        topic: str,
        stance: Literal["for", "against"],
        text: str,
        injected_faults: list[str] | None = None,
        expected_score: int = 100,
        source: Literal["generated", "curated", "user"] = "generated",
        generated_by: str | None = None,
    ) -> "Argument":
        """Create a new argument with a generated ID."""
        return cls(
            id=str(uuid.uuid4())[:8],
            topic=topic,
            stance=stance,
            text=text,
            injected_faults=injected_faults or [],
            expected_score=expected_score,
            source=source,
            generated_by=generated_by,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "stance": self.stance,
            "text": self.text,
            "injected_faults": self.injected_faults,
            "expected_score": self.expected_score,
            "source": self.source,
            "generated_by": self.generated_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Argument":
        return cls(**data)


@dataclass
class EvaluationResult:
    """Result of evaluating an argument."""
    argument_id: str

    # Evaluator configuration
    model: str
    temperature: float
    system_prompt: str  # Name of prompt used (e.g., "default", "strict")

    # Raw results from evaluator
    score: int
    detected_faults: list[str] = field(default_factory=list)
    reasoning: str = ""

    # Computed metrics (vs ground truth)
    score_delta: int = 0  # actual - expected
    true_positives: list[str] = field(default_factory=list)   # Correctly detected
    false_negatives: list[str] = field(default_factory=list)  # Missed faults
    false_positives: list[str] = field(default_factory=list)  # Incorrectly flagged

    # Metadata
    run_id: str = ""
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def compute_metrics(self, argument: Argument) -> None:
        """Compute ground truth comparison metrics."""
        self.score_delta = self.score - argument.expected_score

        injected = set(argument.injected_faults)
        detected = set(self.detected_faults)

        self.true_positives = list(injected & detected)
        self.false_negatives = list(injected - detected)
        self.false_positives = list(detected - injected)

    @property
    def precision(self) -> float:
        """Precision: what fraction of detected faults were actually present."""
        if not self.detected_faults:
            return 1.0 if not self.false_positives else 0.0
        return len(self.true_positives) / len(self.detected_faults)

    @property
    def recall(self) -> float:
        """Recall: what fraction of actual faults were detected."""
        total_actual = len(self.true_positives) + len(self.false_negatives)
        if total_actual == 0:
            return 1.0
        return len(self.true_positives) / total_actual

    def to_dict(self) -> dict:
        return {
            "argument_id": self.argument_id,
            "model": self.model,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
            "score": self.score,
            "detected_faults": self.detected_faults,
            "reasoning": self.reasoning,
            "score_delta": self.score_delta,
            "true_positives": self.true_positives,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "run_id": self.run_id,
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    description: str = ""

    # Matrix variables (all combinations tested)
    models: list[str] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=lambda: [1.0])
    evaluator_prompts: list[str] = field(default_factory=lambda: ["default"])

    # Arguments to evaluate
    argument_ids: list[str] = field(default_factory=list)

    # Repetition for consistency testing
    runs_per_combination: int = 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "models": self.models,
            "temperatures": self.temperatures,
            "evaluator_prompts": self.evaluator_prompts,
            "argument_ids": self.argument_ids,
            "runs_per_combination": self.runs_per_combination,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_data: dict) -> "ExperimentConfig":
        """Create config from YAML experiment definition."""
        matrix = yaml_data.get("matrix", {})
        return cls(
            name=yaml_data["name"],
            description=yaml_data.get("description", ""),
            models=matrix.get("models", []),
            temperatures=matrix.get("temperatures", [1.0]),
            evaluator_prompts=matrix.get("evaluator_prompts", ["default"]),
            argument_ids=yaml_data.get("argument_ids", []),
            runs_per_combination=yaml_data.get("runs_per_combination", 1),
        )


@dataclass
class ExperimentResults:
    """Results from a complete experiment."""
    experiment_id: str
    config: ExperimentConfig
    evaluations: list[EvaluationResult] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None

    @classmethod
    def create(cls, config: ExperimentConfig) -> "ExperimentResults":
        return cls(
            experiment_id=f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
        )

    def add_evaluation(self, evaluation: EvaluationResult) -> None:
        self.evaluations.append(evaluation)

    def mark_complete(self) -> None:
        self.completed_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict(),
            "evaluations": [e.to_dict() for e in self.evaluations],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResults":
        return cls(
            experiment_id=data["experiment_id"],
            config=ExperimentConfig.from_dict(data["config"]),
            evaluations=[EvaluationResult.from_dict(e) for e in data["evaluations"]],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
        )
