import random
import yaml

def random_sample_excluding_indices(items, k, exclude_indices):
    """
    Randomly sample k items from a list, excluding items at the provided indices.

    Args:
        items (list): The list to sample from.
        k (int): Number of items to sample.
        exclude_indices (list or set): Indices to exclude from sampling.

    Returns:
        list: Randomly sampled items.
    """
    exclude_indices = set(exclude_indices)
    available_indices = [i for i in range(len(items)) if i not in exclude_indices]
    if k > len(available_indices):
        raise ValueError("Not enough items to sample after excluding indices.")
    chosen_indices = random.sample(available_indices, k)
    return [items[i] for i in chosen_indices]


class YamlConfig:
    def __init__(self, path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        self._set_attrs(config)

    def _set_attrs(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                value = YamlConfig._from_dict(value)
            setattr(self, key, value)

    @staticmethod
    def _from_dict(d):
        obj = type('YamlConfigSection', (), {})()
        for key, value in d.items():
            if isinstance(value, dict):
                value = YamlConfig._from_dict(value)
            setattr(obj, key, value)
        return obj

    def get(self, key, default=None):
        return getattr(self, key, default)
