#!/usr/bin/env python3
"""Merge SDST, raw SDST, and raw FADANA NanoAOD files event-by-event."""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


TREE = "Events"
RUN = "Event_runNumber"
EVENT = "Event_eventNumber"
EVENT_ALIASES = (EVENT, "Event_evtNumber")

RAW_DROP = ("Event_*", "nGenPart", "GenPart", "GenPart_*")
PREFIXES = {"sdst": "SDST_", "raw_sdst": "RAWSDST_", "raw_fadana": "RAWFADANA_"}


def unique(items):
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def parent_branch(name, keys):
    parts = name.split(".")
    for stop in range(1, len(parts)):
        parent = ".".join(parts[:stop])
        if parent in keys:
            return parent
    return name


def selected_branches(path, raw=False):
    with uproot.open(path) as root_file:
        keys = list(root_file[TREE].keys())

    skip = {RUN, EVENT, "Event_evtNumber"}
    selected = []
    for key in keys:
        if key in skip or (raw and any(fnmatch.fnmatchcase(key, pattern) for pattern in RAW_DROP)):
            continue
        selected.append(parent_branch(key, set(keys)))
    return unique(selected), keys


def event_branch(keys):
    for name in EVENT_ALIASES:
        if name in keys:
            return name
    raise RuntimeError(f"missing {EVENT}")


def read_sample(path, branches, keys):
    event_name = event_branch(keys)
    read_names = [event_name]
    if RUN in keys:
        read_names.append(RUN)
    for branch in branches:
        read_names.append(f"{branch}.*" if any(key.startswith(f"{branch}.") for key in keys) else branch)

    with uproot.open(path) as root_file:
        arrays = root_file[TREE].arrays(unique(read_names), library="ak", how=dict)

    sample = {
        "arrays": arrays,
        "branches": [branch for branch in branches if branch in arrays],
        "event": np.asarray(ak.to_numpy(arrays[event_name]), dtype=np.int64),
        "run": np.asarray(ak.to_numpy(arrays[RUN]), dtype=np.int64) if RUN in arrays else None,
    }
    return sample


def take(sample, indexer):
    return {
        "arrays": {name: array[indexer] for name, array in sample["arrays"].items()},
        "branches": sample["branches"],
        "event": sample["event"][indexer],
        "run": None if sample["run"] is None else sample["run"][indexer],
    }


def keys_for(sample, use_run):
    if use_run:
        return list(zip(sample["run"].astype(int), sample["event"].astype(int), strict=True))
    return list(sample["event"].astype(int))


def align_to(base_keys, sample, use_run):
    index = {key: i for i, key in enumerate(keys_for(sample, use_run))}
    missing = [key for key in base_keys if key not in index]
    if missing:
        raise RuntimeError(f"input is missing events: {missing[:10]}")
    return take(sample, np.array([index[key] for key in base_keys]))


def add_prefixed(output, sample, prefix):
    for branch in sample["branches"]:
        output[f"{prefix}{branch}"] = sample["arrays"][branch]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sdst", required=True, type=Path)
parser.add_argument("--raw-sdst", required=True, type=Path)
parser.add_argument("--raw-fadana", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

sdst_branches, sdst_keys = selected_branches(args.sdst)
raw_sdst_branches, raw_sdst_keys = selected_branches(args.raw_sdst, raw=True)
raw_fadana_branches, raw_fadana_keys = selected_branches(args.raw_fadana, raw=True)

sdst = read_sample(args.sdst, sdst_branches, sdst_keys)
raw_sdst = read_sample(args.raw_sdst, raw_sdst_branches, raw_sdst_keys)
raw_fadana = read_sample(args.raw_fadana, raw_fadana_branches, raw_fadana_keys)

raw_sdst = take(raw_sdst, raw_sdst["event"] != 0)
raw_fadana = take(raw_fadana, raw_fadana["event"] != 0)

use_run = all(sample["run"] is not None for sample in (sdst, raw_sdst, raw_fadana))
use_run = use_run and not any(np.all(sample["run"] == 0) for sample in (sdst, raw_sdst, raw_fadana))
print(f"Matching by ({RUN}, {EVENT})" if use_run else f"Matching by {EVENT}")

base_keys = keys_for(sdst, use_run)
raw_sdst = align_to(base_keys, raw_sdst, use_run)
raw_fadana = align_to(base_keys, raw_fadana, use_run)

output = {
    RUN: sdst["run"] if sdst["run"] is not None else np.zeros(len(sdst["event"]), dtype=np.int64),
    EVENT: sdst["event"],
}
add_prefixed(output, sdst, PREFIXES["sdst"])
add_prefixed(output, raw_sdst, PREFIXES["raw_sdst"])
add_prefixed(output, raw_fadana, PREFIXES["raw_fadana"])

args.output.parent.mkdir(parents=True, exist_ok=True)
with uproot.recreate(args.output) as root_file:
    root_file[TREE] = output

event_count = len(sdst["event"])
print(f"Wrote {args.output} with {event_count} events as RNTuple")
