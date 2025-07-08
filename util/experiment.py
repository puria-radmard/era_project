"""
Experiment Configuration Module

This module contains the configuration dataclass for running experiments with 
HuggingFace model utilities.
"""

from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for model evaluation experiments."""
    
    model_name: str
    save_path: str
    minibatch_size: int
    system_prompt: str
    include_sentence_answers: bool

    mcq_template: str
    mcq_prefiller: str
    sentence_template: str
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """
        Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            ExperimentConfig instance loaded from YAML
            
        Example YAML format:
            model_name: "meta-llama/Llama-2-7b-chat-hf"
            save_path: "./results/experiment_1.json"
            minibatch_size: 4
            system_prompt: "You are an expert assistant."
        """
        with open(yaml_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            yaml_path: Path where to save the YAML configuration file
        """
        config_dict = {
            'model_name': self.model_name,
            'save_path': self.save_path,
            'minibatch_size': self.minibatch_size,
            'system_prompt': self.system_prompt
        }
        
        # Ensure directory exists
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        
        # Ensure save directory exists
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)


# Example usage and default config creation
if __name__ == "__main__":
    # Create a default configuration
    default_config = ExperimentConfig(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        save_path="./results/default_experiment.json",
        minibatch_size=4,
        system_prompt="You are an expert assistant that provides accurate answers."
    )
    
    # Save to YAML
    default_config.to_yaml("./configs/default_config.yaml")
    print("Default configuration saved to ./configs/default_config.yaml")
    
    # Load from YAML
    loaded_config = ExperimentConfig.from_yaml("./configs/default_config.yaml")
    print(f"Loaded config: {loaded_config}")