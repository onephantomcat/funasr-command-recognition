#!/usr/bin/env python3
"""Run the frozen P1 v2_b1 builder on Windows or POSIX.

This adapter leaves ``build_p1_v2_b1.py`` unchanged.  It only supplies the
local dataset path aliases and reloads split manifests in Windows ``spawn``
workers; sample construction, schema, acceptance checks, and finalization all
remain in the frozen builder.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import traceback
from pathlib import Path, PureWindowsPath
from typing import Any

import build_p1_v2_b1 as core


def source_path(row: dict[str, Any], kind: str) -> Path:
    """Resolve the canonical P1 paths plus this workspace's extracted layout."""
    assert core.SOURCE_ROOT is not None
    if kind == "speech":
        value = str(row["audio_path"])
        parts = PureWindowsPath(value).parts
        lower = [part.lower() for part in parts]
        try:
            position = lower.index("data_aishell")
        except ValueError as exc:
            raise ValueError(f"cannot map AISHELL path: {value}") from exc
        suffix = Path(*parts[position:])
        candidates = (
            core.SOURCE_ROOT / "data" / "aishell1" / suffix,
            core.SOURCE_ROOT / "data" / "public" / "aishell1" / "extracted" / suffix,
        )
    else:
        relative = str(row["relative_path"]).replace("\\", "/").lstrip("/")
        if kind == "noise":
            candidates = (
                core.SOURCE_ROOT / "data" / "musan" / "musan" / relative,
                core.SOURCE_ROOT
                / "data"
                / "public"
                / "augmentations"
                / "musan"
                / "extracted"
                / "musan"
                / relative,
            )
        elif kind == "rir":
            candidates = (
                core.SOURCE_ROOT / "data" / "rirs_noises" / "RIRS_NOISES" / relative,
                core.SOURCE_ROOT
                / "data"
                / "public"
                / "augmentations"
                / "rirs_noises"
                / "extracted"
                / "RIRS_NOISES"
                / relative,
            )
        else:
            raise ValueError(f"unknown source kind={kind}")
    return next((path for path in candidates if path.exists()), candidates[0])


# ``core.build_one`` looks this symbol up in the frozen module at runtime.
core.source_path = source_path


def init_worker(source_root: str, stage_root: str, inputs: str) -> None:
    """Initialize globals that are inherited by fork but not by Windows spawn."""
    core.SOURCE_ROOT = Path(source_root)
    core.STAGE_ROOT = Path(stage_root)
    core.load_assets(Path(inputs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(18, (os.cpu_count() or 2) - 2)))
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="build 14 records spanning every formal scenario and asset partition",
    )
    args = parser.parse_args(argv)

    output = args.output.resolve()
    stage = output if args.preflight else output.parent / f"{output.name}.building"
    error_report = output.parent / f"{output.name}_error.json"
    try:
        if output.exists() and not args.preflight:
            raise FileExistsError(f"refusing to overwrite frozen output: {output}")
        stage.mkdir(parents=True, exist_ok=True)
        inputs = args.inputs.resolve()
        core.load_assets(inputs)
        core.SOURCE_ROOT = args.source_root.resolve()
        core.STAGE_ROOT = stage
        input_hashes = {
            path.name: core.sha256_file(path)
            for path in sorted((inputs / "manifests" / "splits").glob("*.jsonl"))
        }

        partial = stage / "manifest.partial.jsonl"
        completed: set[str] = set()
        if partial.exists():
            completed = {str(row["sample_id"]) for row in core.read_jsonl(partial)}

        if args.preflight:
            planned = {
                "train": [0, 35000, 70000, 90000],
                "dev": [0, 3500, 7000, 9000],
                "D_single": [0],
                "D_overlap": [0, 1, 2, 3, 4],
            }
        else:
            planned = {split: list(range(total)) for split, total in core.EXPECTED.items()}

        tasks: list[tuple[str, int]] = []
        for split, indices in planned.items():
            for index in indices:
                sample_id = f"tse_{split.lower()}_{index + 1:08d}"
                if sample_id not in completed:
                    tasks.append((split, index))

        print(
            f"P1_V2_B1_BUILD_BEGIN TASKS={len(tasks)} "
            f"COMPLETED={len(completed)} WORKERS={args.workers}",
            flush=True,
        )
        mode = "a" if partial.exists() else "w"
        context = mp.get_context("spawn" if os.name == "nt" else "fork")
        with partial.open(mode, encoding="utf-8", newline="\n", buffering=1) as journal:
            with context.Pool(
                processes=max(1, args.workers),
                initializer=init_worker,
                initargs=(str(core.SOURCE_ROOT), str(core.STAGE_ROOT), str(inputs)),
            ) as pool:
                done = len(completed)
                for row in pool.imap_unordered(core.build_one, tasks, chunksize=1):
                    journal.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    done += 1
                    if done % 100 == 0 or args.preflight:
                        total = sum(len(indices) for indices in planned.values())
                        print(f"BUILD_PROGRESS={done}/{total}", flush=True)

        if args.preflight:
            print(
                f"P1_V2_B1_PREFLIGHT_STATUS=PASS "
                f"ROWS={len(core.read_jsonl(partial))} OUTPUT={stage}"
            )
            return 0

        core.finalize(stage, partial, input_hashes)
        stage.rename(output)
        print("P1_V2_B1_BUILD_STATUS=PASS")
        print(f"OUTPUT={output}")
        print(f"ROWS={sum(core.EXPECTED.values())}")
        print(f"MANIFEST_SHA256={core.sha256_file(output / 'manifest.jsonl')}")
        print(f"ACCEPTANCE_REPORT={output / 'reports' / 'p1_v2_b1_acceptance.json'}")
        return 0
    except Exception as exc:
        core.write_json(
            error_report,
            {
                "status": "FAIL",
                "time": core.utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "stage": str(stage),
            },
        )
        print("P1_V2_B1_BUILD_STATUS=FAIL")
        print(f"ERROR={exc}")
        print(f"ERROR_REPORT={error_report}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
