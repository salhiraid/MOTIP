# Copyright (c) Ruopeng Gao. All Rights Reserved.

import random
import torch
from math import floor
from torch.utils.data.sampler import Sampler

from .joint_dataset import JointDataset


class NaiveSampler(Sampler):
    def __init__(
            self,
            data_source: JointDataset,
            sample_steps: list,
            sample_lengths: list,
            sample_intervals: list,
            length_per_iteration: int,
            sample_mode: str = "random_interval",
            seed: int = 1025,
            data_weights: dict | None = None,
            samples_per_epoch: int | None = None,
            dataset_mix: dict | None = None,
            max_intervals: dict | None = None,
    ):
        super().__init__()
        self.data_source = data_source
        self.sample_steps = sample_steps
        self.sample_lengths = sample_lengths
        self.sample_intervals = sample_intervals
        self.length_per_iteration = length_per_iteration
        self.sample_mode = sample_mode
        self.seed = seed
        self.data_weights = data_weights
        self.samples_per_epoch = samples_per_epoch      # MOTIP2 App. A: 40k samples per epoch
        # Multi-dataset training (both optional, keyed [dataset][split]):
        #  dataset_mix:   target FRACTION of every epoch's samples drawn from each (dataset, split); needs
        #                 samples_per_epoch. Without it samples are proportional to #legal clips (x integer weights).
        #  max_intervals: per-dataset cap of the random frame interval (low-fps datasets must not skip frames).
        self.dataset_mix = dataset_mix
        self.max_intervals = max_intervals
        if dataset_mix is not None:
            assert samples_per_epoch is not None, "dataset_mix needs samples_per_epoch"
            assert all(f >= 0 for d in dataset_mix.values() for f in d.values()) and \
                sum(f for d in dataset_mix.values() for f in d.values()) > 0, "dataset_mix must be >= 0 and not all 0"
        # Check for these parameters:
        assert len(sample_steps) == len(sample_lengths) == len(sample_intervals), \
            "The lengths of sample_steps, sample_lengths, and sample_intervals should be the same."
        assert sample_mode in ["random_interval"], \
            f"The sample_mode {sample_mode} is not supported."
        for sample_length in sample_lengths:
            if self.length_per_iteration is not None:
                assert sample_length % length_per_iteration == 0, \
                    f"The sample_length {sample_length} should be divisible by {length_per_iteration}."

        # Init the sampling details:
        self.sample_infos = None    # a list of dict: (dataset, split, sequence_name, frame_idxs)
        return

    def prepare_for_epoch(self, epoch: int):
        """
        Decide the sampling details for the current "epoch".
        It should be called at the beginning of each epoch.
        Args:
            epoch: The current epoch.
        """
        # Make sure the random process is the same for each process:
        general_random_state = random.getstate()
        random.seed(self.seed + epoch)      # only for sampling process.
        ############################################################
        sample_infos = []
        pools = {}          # (dataset, split) -> indices into sample_infos
        sample_length: int | None = None
        sample_interval: int | None = None
        # First, calculate the sample length and interval:
        for _ in range(len(self.sample_steps)):
            if epoch >= self.sample_steps[_]:
                sample_length = self.sample_lengths[_]
                sample_interval = self.sample_intervals[_]
        assert sample_length is not None and sample_interval is not None, \
            f"The epoch {epoch} is not supported."
        # Second, get all legal samplings:
        for dataset in self.data_source.annotations:
            for split in self.data_source.annotations[dataset]:
                # Get the weights of the current dataset and split:
                if self.data_weights is not None:
                    if dataset in self.data_weights and split in self.data_weights[dataset]:
                        _weight = self.data_weights[dataset][split]
                    else:
                        _weight = 1
                else:
                    _weight = 1
                assert isinstance(_weight, int), f"The weight {_weight} should be an integer."
                # (float weights: use `dataset_mix`, which is a sampling fraction, instead)
                _cap = sample_interval
                if self.max_intervals is not None and dataset in self.max_intervals \
                        and split in self.max_intervals[dataset]:
                    _cap = max(1, min(sample_interval, int(self.max_intervals[dataset][split])))
                _pool = pools.setdefault((dataset, split), [])
                # Sampling:
                for sequence_name in self.data_source.annotations[dataset][split]:
                    for frame_id in range(self.data_source.sequence_infos[dataset][split][sequence_name]["length"]):
                        _sample_times = _weight
                        for _ in range(_sample_times):
                            if self.data_source.sequence_infos[dataset][split][sequence_name]["is_static"] is True:
                                # static image
                                if self.data_source.ann_is_legals[dataset][split][sequence_name][0]:
                                    _pool.append(len(sample_infos))
                                    sample_infos.append({
                                        "dataset": dataset,
                                        "split": split,
                                        "sequence": sequence_name,
                                        "frame_idxs": [0] * sample_length,
                                    })
                                else:
                                    pass
                            else:   # real-world video
                                if frame_id + sample_length <= self.data_source.sequence_infos[dataset][split][sequence_name]["length"]:
                                    begin_index = frame_id
                                    remain_frames = self.data_source.sequence_infos[dataset][split][sequence_name]["length"] - begin_index - 1
                                    max_interval = floor(remain_frames / (sample_length - 1)) if sample_length > 1 \
                                        else 1
                                    interval = min(random.randint(1, _cap), max_interval)
                                    frame_idxs = [begin_index + interval * _ for _ in range(sample_length)]
                                    _ = torch.tensor(frame_idxs, dtype=torch.int64)
                                    if self.data_source.ann_is_legals[dataset][split][sequence_name][_].all():
                                        # make sure in the sampling sequence,
                                        # all frame's annotation is legal,
                                        # which is friendly for training.
                                        _pool.append(len(sample_infos))
                                        sample_infos.append({
                                            "dataset": dataset,
                                            "split": split,
                                            "sequence": sequence_name,
                                            "frame_idxs": frame_idxs,
                                        })
        total_len = len(sample_infos)
        # Shuffle the samples
        # and only keep the first "total_len // (sample_length / self.length_per_iteration)" samples:
        if self.length_per_iteration is None:
            kept_len = total_len
        else:
            kept_len = int(total_len // (sample_length // self.length_per_iteration))
        # These three lines are copied from the official PyTorch code:
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        indices = torch.randperm(kept_len, generator=g).tolist()
        if self.dataset_mix is not None:
            indices = self._mixed_indices(pools, g)
        elif self.samples_per_epoch is not None:
            # MOTIP2: fixed epoch length; cycle through the (reshuffled-per-epoch) legal samples if too few.
            indices = (indices * (self.samples_per_epoch // max(1, len(indices)) + 1))[:self.samples_per_epoch]
        self.sample_infos = [sample_infos[_] for _ in indices]

        # Recover the random state:
        random.setstate(general_random_state)
        ###########################
        return

    def _mixed_indices(self, pools: dict, g: torch.Generator):
        """samples_per_epoch indices with exactly round(fraction * samples_per_epoch) from every (dataset, split):
        each pool is shuffled and cycled (a pool smaller than its quota is repeated, a larger one is subsampled)."""
        def fraction(key):
            d, sp = key
            return self.dataset_mix.get(d, {}).get(sp, 0.0)
        keys = [k for k in pools if fraction(k) > 0]
        assert keys, "dataset_mix selects no dataset that has legal clips"
        for k in pools:
            if fraction(k) > 0:
                assert len(pools[k]) > 0, f"dataset {k} has weight {fraction(k)} but no legal clip"
        total = sum(fraction(k) for k in keys)
        # largest-remainder rounding so the quotas sum exactly to samples_per_epoch
        raw = {k: self.samples_per_epoch * fraction(k) / total for k in keys}
        quota = {k: int(raw[k]) for k in keys}
        for k in sorted(keys, key=lambda k: raw[k] - quota[k], reverse=True)[: self.samples_per_epoch - sum(quota.values())]:
            quota[k] += 1
        out = []
        for k in keys:
            pool = pools[k]
            order = torch.randperm(len(pool), generator=g).tolist()
            reps = quota[k] // len(pool) + 1
            out += [pool[i] for i in (order * reps)[: quota[k]]]
        perm = torch.randperm(len(out), generator=g).tolist()
        return [out[i] for i in perm]

    def __iter__(self):
        return iter(self.sample_infos)

    def __len__(self):
        return len(self.sample_infos)




