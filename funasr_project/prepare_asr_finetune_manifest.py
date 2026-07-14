"""Create clean, speaker-disjoint ASR manifests from expanded AISHELL audio.

The competition-shaped pos/neg data contains overlapping target speech and is
for end-to-end robustness experiments. Generic ASR fine-tuning instead uses
only clean AISHELL utterances with their original transcripts.
"""
import argparse
import json
from pathlib import Path

import soundfile as sf

from build_external_trainset import read_transcripts, scan_wavs


def write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_records(speakers, texts):
    records = []
    for speaker, wavs in sorted(speakers.items()):
        for wav in wavs:
            text = texts.get(wav.stem, "")
            if not text:
                continue
            info = sf.info(wav)
            if info.samplerate != 16000 or info.frames <= 0:
                continue
            records.append({
                "key": wav.stem,
                "source": str(wav.resolve()),
                "source_len": int(info.frames * 100 / info.samplerate),
                "target": text,
                "target_len": len(text),
                "speaker": speaker,
            })
    return records


def write_split(out, name, records):
    wav_scp = [f"{item['key']} {item['source']}" for item in records]
    text = [f"{item['key']} {item['target']}" for item in records]
    write_lines(out / f"{name}_wav.scp", wav_scp)
    write_lines(out / f"{name}_text", text)
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav-root",
        default="data/public/aishell1/extracted/data_aishell/wav_expanded/train",
    )
    parser.add_argument(
        "--transcript",
        default="data/public/aishell1/extracted/data_aishell/transcript/aishell_transcript_v0.8.txt",
    )
    parser.add_argument("--out", default="data/asr_finetune/aishell1_clean")
    parser.add_argument(
        "--dev-speakers",
        type=int,
        default=3,
        help="Hold out this many complete speakers for development CER.",
    )
    args = parser.parse_args()

    texts = read_transcripts(args.transcript)
    speakers = scan_wavs(args.wav_root)
    usable = {speaker: wavs for speaker, wavs in speakers.items() if wavs}
    speaker_ids = sorted(usable)
    if len(speaker_ids) <= args.dev_speakers:
        raise SystemExit("Need more speakers than --dev-speakers")

    dev_ids = speaker_ids[-args.dev_speakers:]
    train_ids = speaker_ids[:-args.dev_speakers]
    train_records = make_records({speaker: usable[speaker] for speaker in train_ids}, texts)
    dev_records = make_records({speaker: usable[speaker] for speaker in dev_ids}, texts)
    if not train_records or not dev_records:
        raise SystemExit("No aligned train/dev records were found")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_split(out, "train", train_records)
    write_split(out, "dev", dev_records)
    summary = {
        "source": "AISHELL-1 clean expanded audio",
        "train_speakers": train_ids,
        "dev_speakers": dev_ids,
        "train_samples": len(train_records),
        "dev_samples": len(dev_records),
        "format": "FunASR source/target JSONL plus Kaldi-style wav.scp and text",
    }
    (out / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote manifests to {out}")


if __name__ == "__main__":
    main()
