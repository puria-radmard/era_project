import random

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
