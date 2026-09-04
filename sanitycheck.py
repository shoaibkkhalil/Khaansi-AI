# sanity_check.py
import numpy as np
import librosa
import joblib
from huggingface_hub import from_pretrained_keras

print("Loading HeAR...")
model = from_pretrained_keras("google/hear")
serving_fn = model.signatures['serving_default']

print("Loading trained artifacts...")
clf = joblib.load('tb_classifier.joblib')
threshold = joblib.load('tb_threshold.joblib')
imputer = joblib.load('tb_symptom_imputer.joblib')
symptom_cols = joblib.load('tb_symptom_cols.joblib')

TARGET_SR = 16000
TARGET_LEN = 32000

def preprocess_audio(filepath):
    audio, sr = librosa.load(filepath, sr=TARGET_SR, mono=True)
    if len(audio) > TARGET_LEN:
        energy = np.convolve(audio**2, np.ones(TARGET_LEN), mode='valid')
        start = np.argmax(energy)
        audio = audio[start:start+TARGET_LEN]
    else:
        audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
    return audio.astype(np.float32)

print("Running one test clip through the pipeline...")
audio = preprocess_audio('test_clip.wav')  # put any self-test clip here, renamed to this
emb = serving_fn(x=audio[np.newaxis, :])['output_0'].numpy()

# must-ask fields answered "no"; nice-to-have fields unknown -> NaN,
# the trained imputer fills them
placeholder = {c: 0 for c in symptom_cols}
placeholder.update({'sex': 1, 'age': 25, 'reported_cough_dur': 14,
                    'height': np.nan, 'weight': np.nan,
                    'heart_rate': np.nan, 'temperature': np.nan})
symptom_vector = np.array([[placeholder[c] for c in symptom_cols]], dtype=np.float32)
symptom_vector = imputer.transform(symptom_vector)

fused_input = np.hstack([emb, symptom_vector])
prob = clf.predict_proba(fused_input)[0, 1]

print(f"SUCCESS. TB-risk probability: {prob:.3f}, threshold: {threshold:.3f}, flagged: {prob > threshold}")