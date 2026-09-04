"""Tests for Qwen prompts used by the Khaansi AI voice agent.

Run the default mocked checks without using API quota:
    python -m unittest test_qwen

Run the live workspace smoke test explicitly:
    RUN_LIVE_QWEN_SMOKE=1 python test_qwen.py
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

import voice_agent as va


class TestVoiceGrammar(unittest.TestCase):
    def test_static_lines_match_each_persona(self):
        self.assertIn("کرتی ہوں", va.agent_line("greeting", "female"))
        self.assertIn("کرتا ہوں", va.agent_line("greeting", "male"))
        self.assertIn("پوچھوں گی", va.agent_line("thanks_after_cough", "female"))
        self.assertIn("پوچھوں گا", va.agent_line("thanks_after_cough", "male"))
        self.assertIn("کر رہی ہوں", va.agent_line("done_ack", "female"))
        self.assertIn("کر رہا ہوں", va.agent_line("done_ack", "male"))

    def test_female_parse_prompt_includes_feminine_instruction(self):
        response = '{"speak": "میں نے آپ کی بات سمجھ لی ہے۔", "extracted": {"fever": false}}'
        with patch.object(va, "_call", return_value=response) as call:
            data = va.parse_answer(
                "کیا آپ کو بخار آتا ہے؟",
                "نہیں، بخار نہیں ہے۔",
                ["fever"],
                agent_gender="female",
            )

        self.assertIs(data["extracted"]["fever"], False)
        prompt = call.call_args.args[0][0]["content"]
        self.assertIn("selected assistant voice is female", prompt)
        self.assertIn("کرتی ہوں", prompt)
        self.assertIn("پوچھوں گی", prompt)

    def test_male_explanation_prompt_includes_masculine_instruction(self):
        with patch.object(va, "_call", return_value="آپ کی جانچ مکمل ہو گئی ہے۔") as call:
            va.generate_explanation(0.41, 0.317, agent_gender="male")

        prompt = call.call_args.args[0][0]["content"]
        self.assertIn("selected assistant voice is male", prompt)
        self.assertIn("کرتا ہوں", prompt)
        self.assertIn("پوچھوں گا", prompt)

    def test_invalid_gender_fails_before_qwen_call(self):
        with patch.object(va, "_call") as call:
            with self.assertRaises(ValueError):
                va.parse_answer(
                    "کیا آپ کو بخار آتا ہے؟",
                    "نہیں۔",
                    ["fever"],
                    agent_gender="unknown",
                )
        call.assert_not_called()


def run_live_smoke():
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("set DASHSCOPE_API_KEY before running the live Qwen smoke test")

    print("=" * 70)
    print("LIVE TEST 1: female Urdu symptom parsing")
    data = va.parse_answer(
        "کیا آپ کو بخار آتا ہے؟",
        "جی ہاں، دو ہفتے سے کھانسی ہے اور رات کو پسینہ بھی آتا ہے، بخار نہیں ہے۔",
        ["fever", "night_sweats", "reported_cough_dur", "hemoptysis"],
        agent_gender="female",
    )
    print(data)
    assert data["extracted"].get("fever") is False
    assert data["extracted"].get("night_sweats") is True
    assert data["extracted"].get("reported_cough_dur") == 14

    print("=" * 70)
    print("LIVE TEST 2: male Urdu symptom parsing")
    data = va.parse_answer(
        "کیا آپ کو کھانسی میں خون آتا ہے؟",
        "جی نہیں، خون نہیں آتا، بس سکھی خانسی ہے۔",
        ["hemoptysis"],
        agent_gender="male",
    )
    print(data)
    assert data["extracted"].get("hemoptysis") is False

    print("=" * 70)
    print("LIVE TEST 3: female Urdu risk explanation")
    explanation = va.generate_explanation(0.41, 0.317, agent_gender="female")
    print(explanation)


if __name__ == "__main__":
    if os.environ.get("RUN_LIVE_QWEN_SMOKE") == "1":
        run_live_smoke()
        print("\nLive Qwen smoke test passed.")
    else:
        unittest.main()
