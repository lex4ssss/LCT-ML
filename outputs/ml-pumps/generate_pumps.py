import argparse
import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np


def load_generator(path):
    spec = importlib.util.spec_from_file_location('pump_generator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(generator_path, seed):
    module = load_generator(generator_path)
    original = module.generate_pre_fault_sequence
    counter = iter(range(10**9))

    def tagged(*args, **kwargs):
        records = original(*args, **kwargs)
        truth = next(counter)
        for record in records:
            record['truth_trajectory'] = truth
        return records

    module.generate_pre_fault_sequence = tagged
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        frame = module.generate_synthetic_dataset(module.CONFIG)
    frame['truth_trajectory'] = frame['truth_trajectory'].fillna(-1).astype(np.int64)
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generate(args.generator, args.seed).to_csv(args.out, index=False)
