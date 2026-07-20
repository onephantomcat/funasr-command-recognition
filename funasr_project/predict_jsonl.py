"""Run label-free target-speaker ASR for a competition input JSONL.

Expected input fields are: id, 唤醒音频, 唤醒文本, 识别音频. The script never
reads pos/neg folders or 识别文本 labels, so it can run on the mixed test set B.
It writes JSONL records containing only id and content.
"""
import argparse
import json
import time
from pathlib import Path

from asr_demo import build_model, recognize
from speaker_verify import SpeakerGate, build_sv_model


FIELD_ID = "id"
FIELD_WAKE_AUDIO = "\u5524\u9192\u97f3\u9891"
FIELD_COMMAND_AUDIO = "\u8bc6\u522b\u97f3\u9891"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)


def resolve_audio(audio_root, value):
    if not value:
        raise ValueError("Audio path is empty")
    path = Path(value)
    return path if path.is_absolute() else audio_root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Base directory for relative audio paths; defaults to the input JSONL directory.",
    )
    parser.add_argument("--sv-threshold", type=float, default=0.30)
    parser.add_argument("--asr-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    audio_root = Path(args.audio_root) if args.audio_root else input_path.parent
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sv_model = None if args.asr_only else build_sv_model()
    asr_model = build_model(with_punc=False)
    started_at = time.time()
    processed = 0
    accepted = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for line_number, row in read_jsonl(input_path):
            if args.limit is not None and processed >= args.limit:
                break
            if FIELD_ID not in row:
                raise ValueError(f"Input line {line_number} has no id")

            wake_audio = resolve_audio(audio_root, row.get(FIELD_WAKE_AUDIO))
            command_audio = resolve_audio(audio_root, row.get(FIELD_COMMAND_AUDIO))
            if not wake_audio.exists() or not command_audio.exists():
                raise FileNotFoundError(
                    f"Input line {line_number} references missing audio: "
                    f"wake={wake_audio}, command={command_audio}"
                )

            should_accept = True
            if sv_model is not None:
                gate = SpeakerGate(sv_model, threshold=args.sv_threshold)
                gate.enroll(str(wake_audio))
                should_accept, _ = gate.verify(str(command_audio))
            content = recognize(asr_model, str(command_audio))[0] if should_accept else ""
            accepted += int(should_accept)
            out.write(json.dumps({FIELD_ID: row[FIELD_ID], "content": content}, ensure_ascii=False) + "\n")
            processed += 1

    elapsed_ms = round((time.time() - started_at) * 1000, 2)
    print(
        f"Processed {processed} samples, accepted {accepted}, "
        f"inference time {elapsed_ms} ms. Output: {output_path}"
    )


if __name__ == "__main__":
    main()
