"""Disjoint scene-seed splits. Test seeds are used for final reporting only."""
SPLITS = {"train": (0, 100_000), "val": (1_000_000, 1_000_200), "test": (2_000_000, 2_000_500)}


def split_seeds(name):
    lo, hi = SPLITS[name]
    return range(lo, hi)