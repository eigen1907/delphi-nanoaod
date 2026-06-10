#!/usr/bin/env python3
"""Merge SDST, raw SDST, and raw FADANA NanoAOD files event-by-event.

Simple policy:
  1. Drop entries with nGenPart == 0.
  2. Match by (runNumber, eventNumber, nGenPart, summed GenPart four-momentum).
  3. Drop keys that are still duplicated after the nGenPart filter.
"""
from __future__ import annotations

import argparse
import fnmatch
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

TREE = "Events"
META_TREE = "Meta"

RUN = "Event_runNumber"
EVENT = "Event_eventNumber"
EVENT_ALIASES = ("Event_eventNumber", "Event_evtNumber")

RAW_DROP = ("Event_*", "nGenPart", "GenPart", "GenPart_*")

PREFIXES = {
    "sdst": "SDST_",
    "raw_sdst": "RAWSDST_",
    "raw_fadana": "RAWFADANA_",
}

P4_CANDIDATES = ("GenPart_fourMomentum", "GenPart_vector")

ROUND_DECIMALS = 6


def unique(items):
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def parent_branch(name: str, keys: set[str]) -> str:
    parts = name.split(".")
    for stop in range(1, len(parts)):
        parent = ".".join(parts[:stop])
        if parent in keys:
            return parent
    return name


def selected_branches(path: Path, *, raw: bool = False) -> tuple[list[str], list[str]]:
    with uproot.open(path) as root_file:
        keys = list(root_file[TREE].keys())

    key_set = set(keys)
    skip = {RUN, *EVENT_ALIASES}
    selected = []

    for key in keys:
        if key in skip:
            continue
        if raw and any(fnmatch.fnmatchcase(key, pattern) for pattern in RAW_DROP):
            continue
        selected.append(parent_branch(key, key_set))

    return unique(selected), keys


def event_branch(keys: list[str]) -> str:
    for name in EVENT_ALIASES:
        if name in keys:
            return name
    raise RuntimeError(f"missing event number branch. Tried: {EVENT_ALIASES}")


def p4_branch(keys: list[str]) -> str:
    key_set = set(keys)
    for name in P4_CANDIDATES:
        if name in key_set:
            return name
    raise RuntimeError("missing GenPart four-momentum branches")


def read_sample(path: Path, branches: list[str], keys: list[str]):
    event_name = event_branch(keys)
    p4_name = p4_branch(keys)

    read_names = [event_name, RUN, "nGenPart", f"{p4_name}.*"]

    for branch in branches:
        if any(key.startswith(f"{branch}.") for key in keys):
            read_names.append(f"{branch}.*")
        else:
            read_names.append(branch)

    with uproot.open(path) as root_file:
        arrays = root_file[TREE].arrays(unique(read_names), library="ak", how=dict)

    n_genpart = np.asarray(ak.to_numpy(arrays["nGenPart"]), dtype=np.int64)

    # Event-level GenPart four-momentum summary.
    p4 = arrays[p4_name]
    px_sum = np.round(ak.to_numpy(ak.sum(p4.fCoordinates.fX, axis=1)), ROUND_DECIMALS)
    py_sum = np.round(ak.to_numpy(ak.sum(p4.fCoordinates.fY, axis=1)), ROUND_DECIMALS)
    pz_sum = np.round(ak.to_numpy(ak.sum(p4.fCoordinates.fZ, axis=1)), ROUND_DECIMALS)
    e_sum = np.round(ak.to_numpy(ak.sum(p4.fCoordinates.fT, axis=1)), ROUND_DECIMALS)

    output_arrays = {branch: arrays[branch] for branch in branches if branch in arrays}

    return {
        "arrays": output_arrays,
        "branches": list(output_arrays),
        "run": np.asarray(ak.to_numpy(arrays[RUN]), dtype=np.int64),
        "event": np.asarray(ak.to_numpy(arrays[event_name]), dtype=np.int64),
        "n_genpart": n_genpart,
        "px_sum": px_sum,
        "py_sum": py_sum,
        "pz_sum": pz_sum,
        "e_sum": e_sum,
    }


def base_valid_mask(sample) -> np.ndarray:
    return sample["n_genpart"] > 0


def keys_for(sample) -> list[tuple[int, int, int, float, float, float, float]]:
    return [
        (
            int(run),
            int(event),
            int(n_genpart),
            float(px_sum),
            float(py_sum),
            float(pz_sum),
            float(e_sum),
        )
        for run, event, n_genpart, px_sum, py_sum, pz_sum, e_sum in zip(
            sample["run"],
            sample["event"],
            sample["n_genpart"],
            sample["px_sum"],
            sample["py_sum"],
            sample["pz_sum"],
            sample["e_sum"],
            strict=True,
        )
    ]


def unique_valid_mask(sample) -> np.ndarray:
    keys = keys_for(sample)
    base_valid = base_valid_mask(sample)

    counts = Counter(key for key, keep in zip(keys, base_valid, strict=True) if keep)

    return np.asarray(
        [keep and counts[key] == 1 for key, keep in zip(keys, base_valid, strict=True)],
        dtype=bool,
    )


def key_to_index(sample, valid: np.ndarray) -> dict:
    keys = keys_for(sample)
    return {key: i for i, key in enumerate(keys) if valid[i]}


def take(sample, indexer: np.ndarray):
    return {
        "arrays": {name: array[indexer] for name, array in sample["arrays"].items()},
        "branches": sample["branches"],
        "run": sample["run"][indexer],
        "event": sample["event"][indexer],
        "n_genpart": sample["n_genpart"][indexer],
        "px_sum": sample["px_sum"][indexer],
        "py_sum": sample["py_sum"][indexer],
        "pz_sum": sample["pz_sum"][indexer],
        "e_sum": sample["e_sum"][indexer],
    }


def add_prefixed(output: dict, sample, prefix: str) -> None:
    for branch in sample["branches"]:
        output[f"{prefix}{branch}"] = sample["arrays"][branch]


def meta_scalar(value: int):
    return np.asarray([value], dtype=np.int64)


def meta_vector(values):
    return ak.Array([values])


def duplicate_count_after_filter(sample) -> int:
    keys = keys_for(sample)
    base_valid = base_valid_mask(sample)
    counts = Counter(key for key, keep in zip(keys, base_valid, strict=True) if keep)
    return sum(count - 1 for count in counts.values() if count > 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdst", required=True, type=Path)
    parser.add_argument("--raw-sdst", required=True, type=Path)
    parser.add_argument("--raw-fadana", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sdst_branches, sdst_keys = selected_branches(args.sdst, raw=False)
    raw_sdst_branches, raw_sdst_keys = selected_branches(args.raw_sdst, raw=True)
    raw_fadana_branches, raw_fadana_keys = selected_branches(args.raw_fadana, raw=True)

    sdst = read_sample(args.sdst, sdst_branches, sdst_keys)
    raw_sdst = read_sample(args.raw_sdst, raw_sdst_branches, raw_sdst_keys)
    raw_fadana = read_sample(args.raw_fadana, raw_fadana_branches, raw_fadana_keys)

    print(
        "Matching by "
        "(Event_runNumber, Event_eventNumber, nGenPart, summed GenPart four-momentum)"
    )

    sdst_valid = unique_valid_mask(sdst)
    raw_sdst_valid = unique_valid_mask(raw_sdst)
    raw_fadana_valid = unique_valid_mask(raw_fadana)

    raw_sdst_index = key_to_index(raw_sdst, raw_sdst_valid)
    raw_fadana_index = key_to_index(raw_fadana, raw_fadana_valid)

    sdst_keys_all = keys_for(sdst)
    sdst_merged = np.asarray(
        [
            valid and key in raw_sdst_index and key in raw_fadana_index
            for key, valid in zip(sdst_keys_all, sdst_valid, strict=True)
        ],
        dtype=bool,
    )

    output_sdst_index = np.flatnonzero(sdst_merged)
    output_keys = [sdst_keys_all[i] for i in output_sdst_index]

    output_raw_sdst_index = np.asarray([raw_sdst_index[key] for key in output_keys], dtype=np.int64)
    output_raw_fadana_index = np.asarray([raw_fadana_index[key] for key in output_keys], dtype=np.int64)

    raw_sdst_merged = np.zeros(len(raw_sdst["event"]), dtype=bool)
    raw_fadana_merged = np.zeros(len(raw_fadana["event"]), dtype=bool)
    raw_sdst_merged[output_raw_sdst_index] = True
    raw_fadana_merged[output_raw_fadana_index] = True

    sdst_out = take(sdst, output_sdst_index)
    raw_sdst_out = take(raw_sdst, output_raw_sdst_index)
    raw_fadana_out = take(raw_fadana, output_raw_fadana_index)

    output = {
        RUN: sdst_out["run"],
        EVENT: sdst_out["event"],
    }
    add_prefixed(output, sdst_out, PREFIXES["sdst"])
    add_prefixed(output, raw_sdst_out, PREFIXES["raw_sdst"])
    add_prefixed(output, raw_fadana_out, PREFIXES["raw_fadana"])

    meta = {
        "n_sdst_input": meta_scalar(len(sdst["event"])),
        "n_sdst_invalid_genpart0": meta_scalar(np.count_nonzero(sdst["n_genpart"] == 0)),
        "n_sdst_duplicate_after_gen_filter": meta_scalar(duplicate_count_after_filter(sdst)),
        "n_sdst_valid": meta_scalar(np.count_nonzero(sdst_valid)),
        "n_sdst_merged": meta_scalar(np.count_nonzero(sdst_merged)),
        "n_raw_sdst_input": meta_scalar(len(raw_sdst["event"])),
        "n_raw_sdst_invalid_genpart0": meta_scalar(np.count_nonzero(raw_sdst["n_genpart"] == 0)),
        "n_raw_sdst_duplicate_after_gen_filter": meta_scalar(duplicate_count_after_filter(raw_sdst)),
        "n_raw_sdst_valid": meta_scalar(np.count_nonzero(raw_sdst_valid)),
        "n_raw_sdst_merged": meta_scalar(np.count_nonzero(raw_sdst_merged)),
        "n_raw_fadana_input": meta_scalar(len(raw_fadana["event"])),
        "n_raw_fadana_invalid_genpart0": meta_scalar(np.count_nonzero(raw_fadana["n_genpart"] == 0)),
        "n_raw_fadana_duplicate_after_gen_filter": meta_scalar(duplicate_count_after_filter(raw_fadana)),
        "n_raw_fadana_valid": meta_scalar(np.count_nonzero(raw_fadana_valid)),
        "n_raw_fadana_merged": meta_scalar(np.count_nonzero(raw_fadana_merged)),
        "sdst_run_number": meta_vector(sdst["run"]),
        "sdst_event_number": meta_vector(sdst["event"]),
        "sdst_n_genpart": meta_vector(sdst["n_genpart"]),
        "sdst_event_valid": meta_vector(sdst_valid),
        "sdst_event_merged": meta_vector(sdst_merged),
        "raw_sdst_run_number": meta_vector(raw_sdst["run"]),
        "raw_sdst_event_number": meta_vector(raw_sdst["event"]),
        "raw_sdst_n_genpart": meta_vector(raw_sdst["n_genpart"]),
        "raw_sdst_event_valid": meta_vector(raw_sdst_valid),
        "raw_sdst_event_merged": meta_vector(raw_sdst_merged),
        "raw_fadana_run_number": meta_vector(raw_fadana["run"]),
        "raw_fadana_event_number": meta_vector(raw_fadana["event"]),
        "raw_fadana_n_genpart": meta_vector(raw_fadana["n_genpart"]),
        "raw_fadana_event_valid": meta_vector(raw_fadana_valid),
        "raw_fadana_event_merged": meta_vector(raw_fadana_merged),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(args.output) as root_file:
        root_file[TREE] = output
        root_file[META_TREE] = meta

    print(f"Wrote {args.output} with {len(sdst_out['event'])} events")
    print(
        "Merge QA: "
        f"SDST {np.count_nonzero(sdst_merged)}/{np.count_nonzero(sdst_valid)}, "
        f"RAWSDST {np.count_nonzero(raw_sdst_merged)}/{np.count_nonzero(raw_sdst_valid)}, "
        f"RAWFADANA {np.count_nonzero(raw_fadana_merged)}/{np.count_nonzero(raw_fadana_valid)} merged"
    )
    print(
        "Invalid nGenPart==0: "
        f"SDST {np.count_nonzero(sdst['n_genpart'] == 0)}, "
        f"RAWSDST {np.count_nonzero(raw_sdst['n_genpart'] == 0)}, "
        f"RAWFADANA {np.count_nonzero(raw_fadana['n_genpart'] == 0)}"
    )


if __name__ == "__main__":
    main()
