import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import os

# --- Core Audio Fingerprinting Functions ---
def get_peaks(S_db):
    local_max = maximum_filter(S_db, size=15) == S_db
    threshold = np.max(S_db) - 30
    return local_max & (S_db > threshold)

def create_hashes(peaks, target_zone_size=5, time_delay=20):
    hashes = {}
    peaks = sorted(peaks, key=lambda x: x[0])
    for i in range(len(peaks)):
        for j in range(1, target_zone_size):
            if i + j < len(peaks):
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]
                if 0 < (t2 - t1) < time_delay:
                    hashes[(f1, f2, t2 - t1)] = t1
    return hashes

def process_audio(file_bytes):
    y, sr = librosa.load(file_bytes, sr=22050)
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    peaks_mask = get_peaks(S_db)
    
    freq_idx, time_idx = np.where(peaks_mask)
    hashes = create_hashes(list(zip(time_idx, freq_idx)))
    return hashes, S_db, sr, time_idx, freq_idx

def match_query(q_hashes, db_dict):
    best_match = "none"
    max_score = 0
    best_offsets = []

    for song_name, db_hashes in db_dict.items():
        offsets = []
        for h_key, t1_q in q_hashes.items():
            if h_key in db_hashes:
                t1_db = db_hashes[h_key]
                offsets.append(t1_db - t1_q)
        
        if offsets:
            offset_counts = pd.Series(offsets).value_counts()
            score = offset_counts.iloc[0]
            if score > max_score and score > 10: 
                max_score = score
                best_match = song_name
                best_offsets = offsets

    return best_match, max_score, best_offsets

# --- Streamlit UI ---
st.set_page_config(page_title="Sonic Signatures", layout="wide", page_icon="🎵")

st.title("🎵 EE200: Sonic Signatures")
st.markdown("Build an audio database and identify query clips using spectrogram constellation matching.")

# Initialize session state for the database
if 'database' not in st.session_state:
    st.session_state.database = {}

# ==========================================
# STEP 1: DATABASE BUILDER
# ==========================================
with st.container(border=True):
    st.header("🗄️ 1. Build Database")
    st.caption("Upload the full-length reference tracks here.")
    
    db_files = st.file_uploader("Upload full song MP3s/WAVs", type=["mp3

