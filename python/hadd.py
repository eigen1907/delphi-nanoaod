#!/usr/bin/env python3
"""Concatenate merged NanoAOD files with uproot, preserving Events and Meta trees."""
from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import uproot


def tree_exists(root_file, tree_name: str) -> bool:
    return tree_name in root_file or f"{tree_name};1" in root_file


def concatenate_tree(paths: list[Path], tree_name: str, *, required: bool = True):
    chunks = []
    branches = None
    used_paths = []

    for path in paths:
        with uproot.open(path) as root_file:
            if not tree_exists(root_file, tree_name):
                if required:
                    raise RuntimeError(f"{path} does not contain tree '{tree_name}'")
                print(f"Warning: {path} does not contain tree '{tree_name}', skipping it")
                continue

            arrays = root_file[tree_name].arrays(branches, library="ak", how=dict)

        if branches is None:
            branches = list(arrays)
        chunks.append(arrays)
        used_paths.append(path)

    if not chunks:
        return None, []

    output = {
        branch: ak.concatenate([chunk[branch] for chunk in chunks], axis=0)
        for branch in branches
    }
    return output, used_paths


def n_entries(tree_arrays: dict[str, ak.Array]) -> int:
    if not tree_arrays:
        return 0
    return len(next(iter(tree_arrays.values())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--tree", default="Events", help="event tree name, default: Events")
    parser.add_argument("--meta-tree", default="Meta", help="metadata/QA tree name, default: Meta")
    parser.add_argument("--no-meta", action="store_true", help="do not write the Meta tree")
    parser.add_argument(
        "--allow-missing-meta",
        action="store_true",
        help="skip inputs without Meta instead of failing",
    )
    args = parser.parse_args()

    events, event_paths = concatenate_tree(args.inputs, args.tree, required=True)

    meta = None
    meta_paths = []
    if not args.no_meta:
        meta, meta_paths = concatenate_tree(
            args.inputs,
            args.meta_tree,
            required=not args.allow_missing_meta,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(args.output) as root_file:
        root_file[args.tree] = events
        if meta is not None:
            root_file[args.meta_tree] = meta

    print(
        f"Wrote {args.output} with {n_entries(events)} events "
        f"from {len(event_paths)} file(s)"
    )
    if not args.no_meta:
        if meta is not None:
            print(
                f"Wrote {args.meta_tree} with {n_entries(meta)} entries "
                f"from {len(meta_paths)} file(s)"
            )
        else:
            print(f"No {args.meta_tree} tree was written")


if __name__ == "__main__":
    main()
