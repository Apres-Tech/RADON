"""Held-out evaluation protocol defining disjoint seed bands for training and testing."""

from typing import List

# Disjoint seed bands
TRAIN_SEED_MIN = 1
TRAIN_SEED_MAX = 10000
TRAIN_EVAL_SEEDS: List[int] = list(range(1, 21))  # 20 fixed training-distribution seeds for eval checkpoints

# Held-out test seeds that training NEVER sees
TEST_SEED_MIN = 50000
TEST_EVAL_SEEDS: List[int] = list(range(50000, 50020))  # 20 fixed held-out seeds for eval checkpoints
BENCHMARK_TEST_SEEDS: List[int] = list(range(50000, 50100))  # 100 held-out seeds for comprehensive benchmark
