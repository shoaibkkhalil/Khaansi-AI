"""
Khaansi AI — local Streamlit sanity/demo app
Wraps the validated HeAR-embedding + symptom-classifier pipeline
(same logic as sanity_check.py) in a simple UI.
"""

import io
import numpy as np
import librosa
import joblib
import streamlit as st
from huggingface_hub import from_pretrained_keras

import voice_agent as va

# ----------------------------
# Step 2: Cached loading
# ----------------------------
# @st.cache_resource makes sure the HeAR model and the joblib artifacts
# are loaded ONCE per session, not on every widget interaction/rerun.
# Without this, Streamlit reloads a multi-hundred-MB model every time
# you touch a slider — that's the #1 thing that makes ML apps in
# Streamlit feel broken if you skip it.

@st.cache_resource
def load_hear_model():
    model = from_pretrained_keras("google/hear")
    return model.signatures["serving_default"]

@st.cache_resource
def load_artifacts():
    clf = joblib.load("tb_classifier.joblib")
    threshold = joblib.load("tb_threshold.joblib")
    imputer = joblib.load("tb_symptom_imputer.joblib")
    symptom_cols = joblib.load("tb_symptom_cols.joblib")
    return clf, threshold, imputer, symptom_cols


TARGET_SR = 16000
TARGET_LEN = 32000


# ----------------------------
# Step 7: Robust audio preprocessing
# ----------------------------
# st.audio_input returns an UploadedFile-like object (BytesIO under the
# hood), usually in WAV or webm/ogg container depending on browser.
# librosa.load can read file-like objects directly as long as the right
# backend (soundfile / audioread + ffmpeg) can decode the container —
# that's why ffmpeg is in your install list. If this step throws, it's
# almost always a codec issue, not a logic issue.

def preprocess_audio(file_like) -> np.ndarray:
    audio, sr = librosa.load(file_like, sr=TARGET_SR, mono=True)
    if len(audio) > TARGET_LEN:
        # pick the loudest TARGET_LEN-sample window (likely the cough itself)
        energy = np.convolve(audio ** 2, np.ones(TARGET_LEN), mode="valid")
        start = np.argmax(energy)
        audio = audio[start:start + TARGET_LEN]
    else:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    return audio.astype(np.float32)


def get_embedding(serving_fn, audio: np.ndarray) -> np.ndarray:
    return serving_fn(x=audio[np.newaxis, :])["output_0"].numpy()


# ----------------------------
# App UI
# ----------------------------

st.set_page_config(page_title="Khaansi AI — Test App", page_icon="🩺")
st.title("🩺 Khaansi AI — Local Test App")
st.caption(
    "Screening aid only — this is a hackathon prototype, "
    "not a medical diagnosis."
)

with st.spinner("Loading model and artifacts..."):
    serving_fn = load_hear_model()
    clf, threshold, imputer, symptom_cols = load_artifacts()

# ----------------------------
# Conversational voice mode (Urdu)
# ----------------------------

VOICE_OPTIONS = {
    "Uzma (female voice)": {"tts_voice": "ur-PK-UzmaNeural", "gender": "female"},
    "Asad (male voice)": {"tts_voice": "ur-PK-AsadNeural", "gender": "male"},
}
MODE_VOICE = "🎙️ Conversational voice (Urdu)"
MODE_MANUAL = "📝 Manual form (English)"


def init_voice_state(profile: dict):
    vs = st.session_state
    greeting = va.agent_line("greeting", profile["gender"])
    if "v_chat" not in vs:
        vs.v_chat = [{"role": "assistant", "text": greeting, "voice": profile["tts_voice"]}]
        vs.v_qa_index = 0
        vs.v_answers = {}
        vs.v_agent_line = greeting
        vs.v_agent_voice = profile["tts_voice"]
        vs.v_cough_bytes = None
        vs.v_cough_emb = None
        vs.v_result = None
        vs.v_retry = 0
        vs.v_played = None
    elif vs.v_cough_bytes is None and vs.v_agent_voice != profile["tts_voice"]:
        vs.v_chat = [{"role": "assistant", "text": greeting, "voice": profile["tts_voice"]}]
        vs.v_agent_line = greeting
        vs.v_agent_voice = profile["tts_voice"]
        vs.v_played = None


def run_voice_mode():
    profile = VOICE_OPTIONS[st.selectbox("Assistant voice", list(VOICE_OPTIONS), key="v_voice")]
    init_voice_state(profile)
    vs = st.session_state
    voice = profile["tts_voice"]
    agent_gender = profile["gender"]

    if not va.llm_available():
        st.warning(
            "Voice mode needs a DASHSCOPE_API_KEY (set it in a .env file — "
            "see .env.example). The manual form still works."
        )
        st.stop()

    # --- chat transcript with per-line replay ---
    for msg in vs.v_chat:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
            st.write(msg["text"])
            if msg["role"] == "assistant":
                st.audio(va.synthesize(msg["text"], msg.get("voice", voice)), format="audio/mp3")

    # --- autoplay the latest agent line once (replay stays above) ---
    latest_key = (vs.get("v_agent_voice", voice), vs.v_agent_line)
    if vs.v_agent_line and vs.v_played != latest_key:
        st.audio(va.synthesize(vs.v_agent_line, latest_key[0]), format="audio/mp3", autoplay=True)
        vs.v_played = latest_key

    # --- Step 1: cough recording ---
    if vs.v_cough_bytes is None:
        st.subheader("Step 1 — Record your cough")
        cough = st.audio_input("Tap to record a cough (a few seconds is enough)", key="cough_rec")
        if cough is not None:
            with st.spinner("Analyzing cough..."):
                vs.v_cough_bytes = cough.getvalue()
                audio = preprocess_audio(io.BytesIO(vs.v_cough_bytes))
                vs.v_cough_emb = get_embedding(serving_fn, audio)
            line = va.agent_line("thanks_after_cough", agent_gender) + " " + va.MUST_ASK[0][1]
            vs.v_chat.append({"role": "assistant", "text": line, "voice": voice})
            vs.v_agent_line = line
            vs.v_agent_voice = voice
            st.rerun()
        return

    # --- Step 2: symptom Q&A ---
    if vs.v_result is None:
        st.subheader("Step 2 — Symptom questions")

        if vs.get("v_error"):
            st.error(vs.v_error)
            vs.v_error = None

        # skip questions the patient already answered (possibly early,
        # if one spoken answer covered several fields)
        while vs.v_qa_index < len(va.MUST_ASK) and va.field_answered(
            vs.v_answers, va.MUST_ASK[vs.v_qa_index][0]
        ):
            vs.v_qa_index += 1
            vs.v_retry = 0

        if vs.v_qa_index < len(va.MUST_ASK):
            field, question = va.MUST_ASK[vs.v_qa_index]
            st.caption(f"سوال {vs.v_qa_index + 1}/{len(va.MUST_ASK)}: {question}")

            with st.expander("Extracted symptoms so far"):
                st.write(vs.v_answers or "—")

            ans = st.audio_input(
                "Tap to record your answer", key=f"ans_{vs.v_qa_index}_{vs.v_retry}"
            )
            if ans is not None:
                with st.spinner("Listening (Whisper STT)..."):
                    text = va.transcribe(ans.getvalue())
                vs.v_chat.append({"role": "user", "text": text if text else "…"})

                unanswered = [f for f, _ in va.MUST_ASK if not va.field_answered(vs.v_answers, f)]
                ack = None
                if text:
                    try:
                        with st.spinner("Thinking (Qwen)..."):
                            out = va.parse_answer(
                                question, text, unanswered, agent_gender=agent_gender
                            )
                        for k, v in out["extracted"].items():
                            if v is not None:
                                vs.v_answers[k] = v
                        ack = out["speak"]
                    except Exception as e:
                        vs.v_error = f"Qwen call failed: {e}"
                        vs.v_retry += 1
                        st.rerun()

                if va.field_answered(vs.v_answers, field) or vs.v_retry >= 1:
                    # answered, or second failed attempt — move on
                    vs.v_retry = 0
                    vs.v_qa_index += 1
                    while vs.v_qa_index < len(va.MUST_ASK) and va.field_answered(
                        vs.v_answers, va.MUST_ASK[vs.v_qa_index][0]
                    ):
                        vs.v_qa_index += 1
                    next_q = (
                        va.MUST_ASK[vs.v_qa_index][1]
                        if vs.v_qa_index < len(va.MUST_ASK)
                        else None
                    )
                    line = ((ack + " ") if ack else "") + (
                        next_q or va.agent_line("done_ack", agent_gender)
                    )
                else:
                    vs.v_retry = 1
                    line = va.agent_line("retry_prefix", agent_gender) + question

                vs.v_chat.append({"role": "assistant", "text": line, "voice": voice})
                vs.v_agent_line = line
                vs.v_agent_voice = voice
                st.rerun()
            return

        # --- all questions done: run the screening model ---
        with st.spinner("Running screening model..."):
            values = va.build_symptom_values(vs.v_answers)
            vector = np.array([[values[c] for c in symptom_cols]], dtype=np.float32)
            vector = imputer.transform(vector)
            fused = np.hstack([vs.v_cough_emb, vector])
            prob = float(clf.predict_proba(fused)[0, 1])
            explanation = va.generate_explanation(
                prob, threshold, agent_gender=agent_gender
            )
        vs.v_result = {"prob": prob, "flagged": prob > threshold, "values": values}
        vs.v_chat.append({"role": "assistant", "text": explanation, "voice": voice})
        vs.v_agent_line = explanation
        vs.v_agent_voice = voice
        st.rerun()

    # --- Step 3: result ---
    res = vs.v_result
    st.subheader("Step 3 — Screening result")
    st.metric("TB-risk probability", f"{res['prob']:.1%}")
    st.caption(f"Decision threshold: {threshold:.3f}")
    if res["flagged"]:
        st.warning(
            "⚠️ Flagged as elevated risk. This is a screening signal only — "
            "please consult a healthcare professional for evaluation."
        )
    else:
        st.success(
            "No elevated risk signal detected. This is a screening aid, "
            "not a diagnosis."
        )

    with st.expander("Symptom vector used by the model", expanded=False):
        st.table(
            {
                "field": list(symptom_cols),
                "value": [
                    ("— (imputer filled)" if np.isnan(res["values"][c]) else res["values"][c])
                    for c in symptom_cols
                ],
            }
        )

    if st.button("🔄 Start over"):
        for k in [
            "v_chat", "v_qa_index", "v_answers", "v_agent_line", "v_agent_voice",
            "v_cough_bytes", "v_cough_emb", "v_result", "v_retry", "v_played",
        ]:
            st.session_state.pop(k, None)
        st.rerun()


mode = st.radio("Mode", [MODE_VOICE, MODE_MANUAL], horizontal=True)

if mode == MODE_VOICE:
    run_voice_mode()
    st.stop()

# ----------------------------
# Manual mode (fallback)
# ----------------------------

# ----------------------------
# Step 3: Audio capture
# ----------------------------
st.subheader("1. Provide a cough recording")

input_mode = st.radio(
    "Input method", ["Record with microphone", "Upload an audio file"], horizontal=True
)

audio_value = None
if input_mode == "Record with microphone":
    audio_value = st.audio_input("Tap to record a cough (a few seconds is enough)")
else:
    audio_value = st.file_uploader(
        "Upload a WAV/MP3/AAC/etc. audio file", type=["wav", "mp3", "m4a", "aac", "ogg", "flac"]
    )

# ----------------------------
# Step 4: Symptom form
# ----------------------------
st.subheader("2. Symptom details")

with st.form("symptom_form"):
    col1, col2 = st.columns(2)

    with col1:
        sex = st.selectbox("Sex", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        height = st.number_input(
            "Height (cm) — optional", min_value=50, max_value=250, value=None
        )
        weight = st.number_input(
            "Weight (kg) — optional", min_value=10, max_value=200, value=None
        )
        reported_cough_dur = st.number_input(
            "Cough duration (days)", min_value=0, max_value=365, value=0
        )
        heart_rate = st.number_input(
            "Heart rate (bpm) — optional", min_value=30, max_value=220, value=None
        )
        temperature = st.number_input(
            "Temperature (°C) — optional", min_value=30.0, max_value=45.0,
            value=None, step=0.1
        )
        smoke_lweek = st.checkbox("Smoked in the last week")

    with col2:
        tb_prior = st.checkbox("Prior TB diagnosis")
        tb_prior_type = st.selectbox(
            "If prior TB — type", ["N/A", "Pulmonary", "Extrapulmonary", "Unknown"]
        )
        hemoptysis = st.checkbox("Coughing up blood (hemoptysis)")
        weight_loss = st.checkbox("Unexplained weight loss")
        fever = st.checkbox("Fever")
        night_sweats = st.checkbox("Night sweats")

    submitted = st.form_submit_button("Analyze")

# ----------------------------
# Step 5: Inference on submit
# ----------------------------
if submitted:
    if audio_value is None:
        st.error("Please record a cough before analyzing.")
        st.stop()

    with st.spinner("Running inference..."):
        # --- audio branch ---
        audio_bytes = io.BytesIO(audio_value.getvalue())
        audio = preprocess_audio(audio_bytes)
        emb = get_embedding(serving_fn, audio)

        # --- symptom branch ---
        # Build the dict first, in whatever order is convenient, then
        # reindex strictly by symptom_cols before turning it into an
        # array — this guarantees the column order matches what the
        # classifier was trained on, regardless of how the form is laid
        # out above.
        # Optional fields left empty are passed as NaN so the trained
        # imputer fills them with learned values — never hardcode defaults.
        values = {
            "sex": 1 if sex == "Male" else 0,
            "age": age,
            "height": np.nan if height is None else height,
            "weight": np.nan if weight is None else weight,
            "reported_cough_dur": reported_cough_dur,
            "tb_prior": 1 if tb_prior else 0,
            "tb_prior_Pul": 1 if tb_prior_type == "Pulmonary" else 0,
            "tb_prior_Extrapul": 1 if tb_prior_type == "Extrapulmonary" else 0,
            "tb_prior_Unknown": 1 if tb_prior_type == "Unknown" else 0,
            "hemoptysis": 1 if hemoptysis else 0,
            "heart_rate": np.nan if heart_rate is None else heart_rate,
            "temperature": np.nan if temperature is None else temperature,
            "weight_loss": 1 if weight_loss else 0,
            "smoke_lweek": 1 if smoke_lweek else 0,
            "fever": 1 if fever else 0,
            "night_sweats": 1 if night_sweats else 0,
        }
        symptom_vector = np.array(
            [[values[c] for c in symptom_cols]], dtype=np.float32
        )
        symptom_vector = imputer.transform(symptom_vector)

        # --- fuse + predict ---
        fused_input = np.hstack([emb, symptom_vector])
        prob = clf.predict_proba(fused_input)[0, 1]
        flagged = prob > threshold

    # ----------------------------
    # Step 6: Display result
    # ----------------------------
    st.subheader("3. Result")
    st.metric("TB-risk probability", f"{prob:.1%}")
    st.caption(f"Decision threshold: {threshold:.3f}")

    if flagged:
        st.warning(
            "⚠️ Flagged as elevated risk. This is a screening signal only — "
            "please consult a healthcare professional for evaluation."
        )
    else:
        st.success(
            "No elevated risk signal detected. This is a screening aid, "
            "not a diagnosis."
        )
