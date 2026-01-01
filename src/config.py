"""Configuration loading utilities."""

import os
from pathlib import Path
from dataclasses import dataclass, field
import yaml


# Default config directory relative to project root
CONFIG_DIR = Path(__file__).parent.parent / "config"


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    id: str
    provider: str
    display_name: str
    model_id: str  # Actual model ID to pass to API
    default_temperature: float = 1.0
    max_tokens: int = 2000


@dataclass
class FaultConfig:
    """Configuration for a single fault type."""
    id: str
    category: str
    description: str
    severity: int
    generation_hint: str = ""


@dataclass
class TopicConfig:
    """Configuration for a debate topic."""
    id: str
    title: str
    category: str
    description: str = ""


@dataclass
class Config:
    """Main configuration container."""
    models: dict[str, ModelConfig] = field(default_factory=dict)
    faults: dict[str, FaultConfig] = field(default_factory=dict)
    topics: dict[str, TopicConfig] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_models(config_dir: Path = CONFIG_DIR) -> dict[str, ModelConfig]:
    """Load model configurations from models.yaml."""
    models_path = config_dir / "models.yaml"
    if not models_path.exists():
        return {}

    data = load_yaml(models_path)
    models = {}

    for config_id, model_data in data.get("models", {}).items():
        models[config_id] = ModelConfig(
            id=config_id,
            provider=model_data["provider"],
            display_name=model_data.get("display_name", config_id),
            model_id=model_data.get("model_id", config_id),  # Actual API model ID
            default_temperature=model_data.get("default_temperature", 1.0),
            max_tokens=model_data.get("max_tokens", 2000),
        )

    return models


def load_faults(config_dir: Path = CONFIG_DIR) -> dict[str, FaultConfig]:
    """Load fault configurations from faults.yaml."""
    faults_path = config_dir / "faults.yaml"
    if not faults_path.exists():
        return {}

    data = load_yaml(faults_path)
    faults = {}

    for category, category_faults in data.get("faults", {}).items():
        for fault_id, fault_data in category_faults.items():
            faults[fault_id] = FaultConfig(
                id=fault_id,
                category=category,
                description=fault_data["description"],
                severity=fault_data["severity"],
                generation_hint=fault_data.get("generation_hint", ""),
            )

    return faults


def load_topics(config_dir: Path = CONFIG_DIR) -> dict[str, TopicConfig]:
    """Load topic configurations from topics.yaml."""
    topics_path = config_dir / "topics.yaml"
    if not topics_path.exists():
        return {}

    data = load_yaml(topics_path)
    topics = {}

    for category, category_topics in data.get("topics", {}).items():
        for topic_data in category_topics:
            topic_id = topic_data["id"]
            topics[topic_id] = TopicConfig(
                id=topic_id,
                title=topic_data["title"],
                category=category,
                description=topic_data.get("description", ""),
            )

    return topics


def load_prompt(name: str, prompt_type: str, config_dir: Path = CONFIG_DIR) -> str:
    """
    Load a prompt template from file.

    Args:
        name: Prompt name (without extension)
        prompt_type: Either 'generators' or 'evaluators'
        config_dir: Config directory path

    Returns:
        Prompt template string
    """
    prompt_path = config_dir / "prompts" / prompt_type / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")

    with open(prompt_path, "r") as f:
        return f.read()


def load_config(config_dir: Path = CONFIG_DIR) -> Config:
    """Load all configuration files."""
    return Config(
        models=load_models(config_dir),
        faults=load_faults(config_dir),
        topics=load_topics(config_dir),
    )


def get_provider_for_model(model_id: str, config: Config | None = None):
    """
    Get the appropriate provider instance for a model.

    Args:
        model_id: Model ID from config
        config: Config object (loads fresh if None)

    Returns:
        LLMProvider instance
    """
    from .providers import AnthropicProvider, OpenAIProvider, XAIProvider, GoogleProvider

    if config is None:
        config = load_config()

    if model_id not in config.models:
        raise ValueError(f"Unknown model: {model_id}")

    model_config = config.models[model_id]
    provider_name = model_config.provider

    if provider_name == "anthropic":
        return AnthropicProvider()
    elif provider_name == "openai":
        return OpenAIProvider()
    elif provider_name == "xai":
        return XAIProvider()
    elif provider_name == "google":
        return GoogleProvider()
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
