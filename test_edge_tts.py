"""Standalone Edge-TTS Urdu test.

Generates the actual phrases the voice agent will speak, with both
Pakistani Urdu voices, and saves them as mp3 files for manual listening.
"""

import asyncio
import os

import edge_tts

VOICES = ["ur-PK-AsadNeural", "ur-PK-UzmaNeural"]

SAMPLES = {
    "greeting": (
        "السلام علیکم! میں کھانسی اے آئی ہوں۔ "
        "میں آپ کی کھانسی کی آواز سن کر آپ کی صحت کا اندازہ لگا سکتا ہوں۔"
    ),
    "question": (
        "کیا آپ کو بخار آتا ہے؟ اور رات کو پسینہ آتا ہے؟"
    ),
    "low_risk": (
        "آپ کے نتائج میں خطرے کی کوئی خاص علامت نہیں ملی۔ "
        "یہ صرف ایک ابتدائی جانچ ہے، مکمل تشخیص نہیں۔ "
        "اگر کھانسی دو ہفتے سے زیادہ رہے تو ڈاکٹر سے ضرور ملیں۔"
    ),
    "high_risk": (
        "آپ کے نتائج میں کچھ علامات خطرے کی طرف اشارہ کرتی ہیں۔ "
        "براہ کرم تصدیقی ٹیسٹ کے لیے قریبی ہسپتال یا کلینک جائیں۔ "
        "یہ تشخیص نہیں ہے، صرف ایک ابتدائی جانچ ہے۔"
    ),
}

OUT_DIR = "tts_test_out"


async def generate(voice: str, name: str, text: str, path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  [{voice}] {name}: {path} ({size_kb:.0f} KB)")


async def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for voice in VOICES:
        for name, text in SAMPLES.items():
            path = os.path.join(OUT_DIR, f"{voice}_{name}.mp3")
            await generate(voice, name, text, path)
    print("Done. Listen to the files in tts_test_out/ to judge voice quality.")


if __name__ == "__main__":
    asyncio.run(main())
