"""Standalone Whisper Urdu STT test.

Round-trip check: transcribe the Edge-TTS-generated Urdu clips with
faster-whisper and compare against the known source text. If Whisper
can accurately read back what Edge-TTS spoke, the STT leg works.
"""

import glob
import sys
import time

from faster_whisper import WhisperModel

sys.stdout.reconfigure(encoding="utf-8")

MODEL_SIZE = "small"

EXPECTED = {
    "greeting": "السلام علیکم... کھانسی اے آئی... کھانسی کی آواز... صحت کا اندازہ",
    "question": "بخار... رات کو پسینہ",
    "low_risk": "خطرے کی کوئی خاص علامت نہیں... ابتدائی جانچ... دو ہفتے... ڈاکٹر",
    "high_risk": "علامات خطرے... تصدیقی ٹیسٹ... ہسپتال یا کلینک... تشخیص نہیں",
}


def main() -> None:
    t0 = time.time()
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print(f"Model '{MODEL_SIZE}' loaded in {time.time() - t0:.1f}s")

    clips = sorted(glob.glob("tts_test_out/ur-PK-AsadNeural_*.mp3"))
    for path in clips:
        t1 = time.time()
        segments, info = model.transcribe(path, language="ur")
        text = " ".join(s.text.strip() for s in segments)
        elapsed = time.time() - t1
        kind = path.rsplit("\\", 1)[-1].split("_", 1)[1].rsplit(".", 1)[0]
        print(f"\n--- {path} ({elapsed:.1f}s) ---")
        print(f"  heard : {text}")
        print(f"  expect keywords: {EXPECTED[kind]}")

    # also test the one with the female voice to be safe
    extra = "tts_test_out/ur-PK-UzmaNeural_high_risk.mp3"
    segments, info = model.transcribe(extra, language="ur")
    text = " ".join(s.text.strip() for s in segments)
    print(f"\n--- {extra} ---")
    print(f"  heard : {text}")


if __name__ == "__main__":
    main()
