"""Standalone Qwen/DashScope text-only test.

Validates the two LLM jobs the voice agent needs, all in text:
  1. parse_answer: turn an Urdu patient answer (incl. noisy
     Whisper-style text) into structured symptom values + a short
     Urdu acknowledgment.
  2. generate_explanation: turn a hardcoded risk score into a
     plain-language Urdu explanation (never a diagnosis).

Requires DASHSCOPE_API_KEY in the environment or a .env file.
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

import voice_agent as va


def show(label, out):
    print(f"raw: {out}")
    data = va._parse_json_block(out)
    print(f"parsed: {data}")
    return data


def test_answer_parsing():
    print("=" * 70)
    print("TEST 1: parse a clean Urdu multi-answer")
    user_msg = (
        '{"current_question": "کیا آپ کو بخار آتا ہے؟", '
        '"patient_answer": "جی ہاں، دو ہفتے سے کھانسی ہے اور رات کو '
        'پسینہ بھی آتا ہے، بخار نہیں ہے۔", '
        '"fields_to_extract": ["fever", "night_sweats", "reported_cough_dur", "hemoptysis"]}'
    )
    data = show("t1", va._call([
        {"role": "system", "content": va.SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]))
    assert data["extracted"].get("fever") is False
    assert data["extracted"].get("night_sweats") is True
    assert data["extracted"].get("reported_cough_dur") == 14
    assert data["extracted"].get("hemoptysis") is None
    assert isinstance(data["speak"], str) and data["speak"].strip()
    print("PASS")


def test_noisy_whisper_input():
    print("=" * 70)
    print("TEST 2: parse a noisy Whisper-style answer")
    user_msg = (
        '{"current_question": "کیا آپ کو کھانسی میں خون آتا ہے؟", '
        '"patient_answer": "جی نہیں، خون نہیں آتا، بس سکھی خانسی ہے۔", '
        '"fields_to_extract": ["hemoptysis"]}'
    )
    data = show("t2", va._call([
        {"role": "system", "content": va.SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]))
    assert data["extracted"].get("hemoptysis") is False
    print("PASS")


def test_risk_explanation():
    print("=" * 70)
    print("TEST 3: Urdu risk explanation (elevated risk, hardcoded score)")
    out = va.generate_explanation(0.41, 0.317)
    print(f"explanation:\n{out}")
    print("PASS (manual check: tone, no diagnosis claim)")


if __name__ == "__main__":
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: set DASHSCOPE_API_KEY (env var or .env file)")
        sys.exit(1)
    test_answer_parsing()
    test_noisy_whisper_input()
    test_risk_explanation()
    print("\nAll Qwen tests passed.")
