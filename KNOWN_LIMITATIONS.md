# Known limitations

## Conversational Urdu voice mode

The conversational voice interview uses **Qwen Plus** via DashScope for symptom dialogue and answer parsing. This was the only free-trial model available in the Alibaba Cloud account used for development, so the agent can sometimes feel a little dull or produce Urdu responses that use difficult words. Edge-TTS handles the actual speech output, but the text it speaks comes from Qwen, so unusual or complex Urdu vocabulary may not sound fully natural.

If a stronger Qwen variant becomes available on the free tier or a paid key is used, switching the model name in `voice_agent.py` would improve response quality and pronunciation flow.

## Demo audio

The audio files referenced in `demo_test_cases.csv` are not committed to the repository. Judges can use the listed filename as a reference, or simply record their own cough audio for testing. To try different outcomes, adjust the symptom values in the CSV:

- **TB positive case:** long cough duration, fever, night sweats, weight loss, hemoptysis, prior TB, etc.
- **TB negative case:** short cough duration, no fever, no night sweats, no weight loss, no hemoptysis, no prior TB, etc.

Upload whichever audio file you record, copy the matching symptom row into the manual form, and click **Analyze**.
