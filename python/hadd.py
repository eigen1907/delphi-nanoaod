#!/usr/bin/env python3
"""Concatenate RNTuple NanoAOD files with uproot."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import uproot


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("inputs", nargs="+", type=Path)
parser.add_argument("--tree", default="Events")
args = parser.parse_args()

chunks = []
branches = None
for path in args.inputs:
    with uproot.open(path) as root_file:
        arrays = root_file[args.tree].arrays(branches, library="ak", how=dict)
    if branches is None:
        branches = list(arrays)
    chunks.append(arrays)

output = {
    branch: ak.concatenate([chunk[branch] for chunk in chunks], axis=0)
    for branch in branches
}

args.output.parent.mkdir(parents=True, exist_ok=True)
with uproot.recreate(args.output) as root_file:
    root_file[args.tree] = output

print(f"Wrote {args.output} with {len(next(iter(output.values())))} events from {len(args.inputs)} file(s)")
