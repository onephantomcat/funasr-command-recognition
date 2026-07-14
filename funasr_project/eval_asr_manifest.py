"""Evaluate the ASR model on a clean source/target JSONL manifest."""
import argparse
import contextlib
import io
import json
import time
from pathlib import Path

from asr_demo import build_model, recognize
from cer import cer, corpus_cer


def read_records(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def recognize_quiet(model, source):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return recognize(model, source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/asr_finetune/aishell1_clean/dev.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--with-punc", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    records = list(read_records(args.manifest))[args.offset:]
    if args.limit is not None:
        records = records[:args.limit]
    if not records:
        raise SystemExit("No records selected")

    print(f"Loading ASR model (punctuation={args.with_punc})...")
    model = build_model(with_punc=args.with_punc)
    started_at = time.time()
    pairs = []
    details = []
    for index, item in enumerate(records, start=1):
        hyp, elapsed = recognize_quiet(model, item["source"])
        value, ref_len = cer(item["target"], hyp)
        pairs.append((item["target"], hyp))
        details.append({
            "key": item["key"],
            "speaker": item.get("speaker"),
            "ref": item["target"],
            "hyp": hyp,
            "cer": round(value, 4),
            "ref_len": ref_len,
            "asr_latency_sec": round(elapsed, 4),
        })
        if index % 25 == 0 or index == len(records):
            print(f"  processed {index}/{len(records)}")

    total_cer, total_chars = corpus_cer(pairs)
    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "samples": len(records),
        "with_punc": args.with_punc,
        "corpus_cer": round(total_cer, 4),
        "reference_chars": total_chars,
        "mean_asr_latency_sec": round(
            sum(item["asr_latency_sec"] for item in details) / len(details), 4
        ),
        "elapsed_sec": round(time.time() - started_at, 2),
        "details": details,
    }
    out = args.out or str(Path(args.manifest).with_name("asr_eval_report.json"))
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"CER={report['corpus_cer']:.4f} ({total_chars} chars), "
        f"mean_asr_latency={report['mean_asr_latency_sec']:.4f}s"
    )
    print(f"Report saved: {out}")


if __name__ == "__main__":
    main()
