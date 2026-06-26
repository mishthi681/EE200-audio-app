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
    
    db_files = st.file_uploader("Upload full song MP3s/WAVs", type=["mp3", "wav"], accept_multiple_files=True, key="db_upload")
    
    # Only show the process button if files are uploaded
    if db_files:
        if st.button("Process & Index Database", type="primary"):
            with st.spinner("Extracting constellation hashes..."):
                st.session_state.database = {}
                for file in db_files:
                    hashes, _, _, _, _ = process_audio(file)
                    song_name = os.path.splitext(file.name)[0]
                    st.session_state.database[song_name] = hashes
                st.success(f"✅ Successfully indexed {len(st.session_state.database)} songs!")
    
    # Always display the current status of the database
    if st.session_state.database:
        st.info(f"Database currently active with **{len(st.session_state.database)}** indexed tracks.")
    else:
        st.warning("Database is empty. Please upload files and click 'Process & Index Database'.")

st.write("") # Spacer

# ==========================================
# STEP 2: IDENTIFICATION
# ==========================================
with st.container(border=True):
    st.header("🔍 2. Identify Audio")
    
    # Prevent user from trying to identify without a database
    if not st.session_state.database:
        st.error("⚠️ You must build the database in Step 1 before you can identify clips.")
    else:
        mode = st.radio("Select Mode:", ("Single Clip (Visual)", "Batch Process (CSV Export)"), horizontal=True)
        st.divider()

        if mode == "Single Clip (Visual)":
            query_file = st.file_uploader("Upload a short query clip", type=["mp3", "wav"], key="q_upload")
            
            if query_file:
                if st.button("Analyze & Identify", type="primary"):
                    with st.spinner("Running identifier..."):
                        q_hashes, S_db, sr, time_idx, freq_idx = process_audio(query_file)
                        match_name, score, offsets = match_query(q_hashes, st.session_state.database)
                        
                        if match_name != "none":
                            st.success(f"🎶 **Match Found:** {match_name} (Confidence Score: {score})")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Query Spectrogram")
                                fig1, ax1 = plt.subplots()
                                librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz', cmap='magma', ax=ax1)
                                t_bins = librosa.frames_to_time(time_idx, sr=sr, hop_length=512)
                                f_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)[list(freq_idx)]
                                ax1.scatter(t_bins, f_bins, color='cyan', s=10, marker='x', alpha=0.8)
                                st.pyplot(fig1)

                            with col2:
                                st.subheader("Alignment Histogram")
                                fig2, ax2 = plt.subplots()
                                ax2.hist(offsets, bins=50, color='coral', edgecolor='black')
                                ax2.set_xlabel("Time Offset")
                                ax2.set_ylabel("Number of Matches")
                                st.pyplot(fig2)
                        else:
                            st.error("No definitive match found in the database.")

        elif mode == "Batch Process (CSV Export)":
            batch_files = st.file_uploader("Upload multiple query clips", type=["mp3", "wav"], accept_multiple_files=True, key="b_upload")
            
            if batch_files:
                if st.button("Run Batch Identification", type="primary"):
                    with st.spinner("Processing batch..."):
                        results = []
                        for file in batch_files:
                            q_hashes, _, _, _, _ = process_audio(file)
                            match_name, _, _ = match_query(q_hashes, st.session_state.database)
                            filename_no_ext = os.path.splitext(file.name)[0]
                            results.append({"filename": filename_no_ext, "prediction": match_name})
                        
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Download results.csv",
                            data=csv,
                            file_name='results.csv',
                            mime='text/csv',
                        )
