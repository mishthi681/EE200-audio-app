


import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import defaultdict
import glob
import os

# --- 1. Spectrogram & Constellation Extraction ---
def generate_constellation(audio_path, window_length=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=22050)
    D = librosa.stft(y, n_fft=window_length, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # 2D maximum filter to find prominent peaks
    local_max = maximum_filter(S_db, size=15) == S_db
    threshold = np.max(S_db) - 30
    peaks = local_max & (S_db > threshold)

    frequencies, times = np.where(peaks)
    return S_db, sr, hop_length, list(zip(times, frequencies))

def plot_spectrogram_and_peaks(audio_path):
    S_db, sr, hop_length, peaks = generate_constellation(audio_path)
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')

    times, freqs = zip(*peaks)
    time_bins = librosa.frames_to_time(times, sr=sr, hop_length=hop_length)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=(S_db.shape[0]-1)*2)[list(freqs)]

    plt.scatter(time_bins, freq_bins, color='cyan', s=10, marker='x', alpha=0.8)
    plt.title(f'Spectrogram & Constellation: {os.path.basename(audio_path)}')
    plt.show()

# --- 2. Hashing & Database Setup ---
def create_hashes(peaks, target_zone_size=5, time_delay=20):
    hashes = {}
    peaks = sorted(peaks, key=lambda x: x[0])
    for i in range(len(peaks)):
        for j in range(1, target_zone_size):
            if i + j < len(peaks):
                t1, f1 = peaks[i]
                t2, f2 = peaks[i + j]
                delta_t = t2 - t1
                if 0 < delta_t < time_delay:
                    hashes[(f1, f2, delta_t)] = t1
    return hashes

song_database = defaultdict(list)

def build_database_current_folder():
    audio_files = glob.glob("*.mp3") + glob.glob("*.wav")
    for file in audio_files:
        song_name = os.path.splitext(file)[0]
        _, _, _, peaks = generate_constellation(file)
        hashes = create_hashes(peaks)
        for hash_key, t1 in hashes.items():
            song_database[hash_key].append((song_name, t1))
    print(f"Database built! Indexed {len(audio_files)} songs.")

build_database_current_folder()

# Note: Call plot_spectrogram_and_peaks("A Hard Day's Night.mp3") here to generate your report images.



import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import pandas as pd
from collections import defaultdict
import os

# --- CORE LOGIC ---
def generate_constellation(audio_path, window_length=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=22050)
    D = librosa.stft(y, n_fft=window_length, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    local_max = maximum_filter(S_db, size=15) == S_db
    peaks = local_max & (S_db > (np.max(S_db) - 30))
    frequencies, times = np.where(peaks)
    return S_db, sr, hop_length, list(zip(times, frequencies))

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

# --- APP INITIALIZATION ---
st.set_page_config(page_title="Zapptain America App", layout="wide")
st.title("🎵 Sonic Signatures: Audio Fingerprinting")

@st.cache_data
def build_database():
    database = defaultdict(list)
    search_dir = "."
    for file in os.listdir(search_dir):
        if file.endswith(('.mp3', '.wav')):
            song_name = os.path.splitext(file)[0]
            _, _, _, peaks = generate_constellation(os.path.join(search_dir, file))
            hashes = create_hashes(peaks)
            for h_key, t1 in hashes.items():
                database[h_key].append((song_name, t1))
    return database

st.info("Indexing song database... Please wait.")
song_db = build_database()
if len(song_db) > 0:
    st.success(f"Database loaded with {len(song_db)} unique frequency hashes!")

# --- UI MODES ---
mode = st.radio("Select Mode", ("Single-Clip Mode", "Batch Mode"))

if mode == "Single-Clip Mode":
    uploaded_file = st.file_uploader("Upload a query audio clip", type=['mp3', 'wav'])
    if uploaded_file is not None:
        with st.spinner("Analyzing audio..."):
            S_db, sr, hop_length, q_peaks = generate_constellation(uploaded_file)
            q_hashes = create_hashes(q_peaks)

            matches = defaultdict(list)
            for h_key, t_query in q_hashes.items():
                if h_key in song_db:
                    for s_name, t_db in song_db[h_key]:
                        matches[s_name].append(t_db - t_query)

            best_song, best_score, best_offsets = "No Match", 0, []
            for song, offsets in matches.items():
                hist, _ = np.histogram(offsets, bins=50)
                if np.max(hist) > best_score:
                    best_score, best_song, best_offsets = np.max(hist), song, offsets

            st.subheader(f"🎯 Recognized Song: **{best_song}**")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Spectrogram & Constellation**")
                fig, ax = plt.subplots()
                librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma', ax=ax)
                if q_peaks:
                    times, freqs = zip(*q_peaks)
                    t_bins = librosa.frames_to_time(times, sr=sr, hop_length=hop_length)
                    f_bins = librosa.fft_frequencies(sr=sr, n_fft=(S_db.shape[0]-1)*2)[list(freqs)]
                    ax.scatter(t_bins, f_bins, color='cyan', s=10, marker='x', alpha=0.8)
                st.pyplot(fig)

            with col2:
                st.write("**Offset Histogram**")
                if best_offsets:
                    fig2, ax2 = plt.subplots()
                    ax2.hist(best_offsets, bins=50, color='coral', edgecolor='black')
                    ax2.set_xlabel("Time Offset")
                    ax2.set_title(f"Alignment: {best_song}")
                    st.pyplot(fig2)

elif mode == "Batch Mode":
    uploaded_files = st.file_uploader("Upload query clips", type=['mp3', 'wav'], accept_multiple_files=True)
    if st.button("Process Batch") and uploaded_files:
        results = []
        progress_bar = st.progress(0)
        for idx, file in enumerate(uploaded_files):
            _, _, _, q_peaks = generate_constellation(file)
            q_hashes = create_hashes(q_peaks)
            matches = defaultdict(list)
            for h_key, t_query in q_hashes.items():
                if h_key in song_db:
                    for s_name, t_db in song_db[h_key]:
                        matches[s_name].append(t_db - t_query)

            best_song, best_score = "No Match", 0
            for song, offsets in matches.items():
                hist, _ = np.histogram(offsets, bins=50)
                if np.max(hist) > best_score:
                    best_score, best_song = np.max(hist), song

            results.append({"filename": file.name, "prediction": best_song})
            progress_bar.progress((idx + 1) / len(uploaded_files))

        df = pd.DataFrame(results)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download results.csv", data=csv, file_name='results.csv', mime='text/csv')

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# 
# import streamlit as st
# import numpy as np
# import librosa
# # ... [rest of your app code below] ...



