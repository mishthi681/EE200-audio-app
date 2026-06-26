


import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import time
import os

# --- Page Config & Styling ---
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide", initial_sidebar_state="collapsed")
plt.style.use('dark_background') # Match the dark aesthetic of the screenshots

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
    times, freqs = np.where(peaks_mask)
    hashes = create_hashes(list(zip(times, freqs)))
    return hashes, S_db, sr, times, freqs

def match_query(q_hashes, db_dict):
    all_scores = {}
    best_match = "None"
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
            all_scores[song_name] = score
            
            if score > max_score and score > 10: 
                max_score = score
                best_match = song_name
                best_offsets = offsets
        else:
            all_scores[song_name] = 0

    # Sort scores descending
    sorted_scores = dict(sorted(all_scores.items(), key=lambda item: item[1], reverse=True))
    return best_match, max_score, best_offsets, sorted_scores

# --- App Header ---
st.title("🎛️ EE200: Audio Fingerprinting")
st.caption("SIGNALS, SYSTEMS & NETWORKS • PROJECT DEMO")
st.markdown("Index a library of songs as spectrogram fingerprints, then identify any short clip against it.")

# --- Session State ---
if 'database' not in st.session_state:
    st.session_state.database = {}

# --- Tabs Navigation ---
tab1, tab2, tab3 = st.tabs(["❖ LIBRARY", "◎ IDENTIFY", "▤ BATCH"])

# ==========================================
# TAB 1: LIBRARY
# ==========================================
with tab1:
    st.subheader("Database Library")
    db_files = st.file_uploader("Upload full song MP3s for the database", type=["mp3", "wav"], accept_multiple_files=True, key="db_upload")
    
    if db_files and st.button("Index Library"):
        with st.spinner("Extracting constellation hashes..."):
            st.session_state.database = {}
            for file in db_files:
                hashes, _, _, _, _ = process_audio(file)
                song_name = os.path.splitext(file.name)[0]
                st.session_state.database[song_name] = hashes
    
    # Display visually similar library cards
    if st.session_state.database:
        st.write("### IN THE DATABASE")
        cols = st.columns(4)
        col_idx = 0
        for song, hashes in st.session_state.database.items():
            with cols[col_idx % 4]:
                st.info(f"**{song}**\n\n{len(hashes):,} hashes")
            col_idx += 1
    else:
        st.info("Song indexing is managed here. Upload tracks to build the library.")

# ==========================================
# TAB 2: IDENTIFY
# ==========================================
with tab2:
    st.subheader("Identify a clip")
    query_file = st.file_uploader("Upload a short query clip", type=["mp3", "wav"], key="query_upload")
    
    if query_file and st.session_state.database:
        if st.button("Identify", type="primary"):
            start_total = time.time()
            
            # Tracking metrics
            t0 = time.time()
            q_hashes, S_db, sr, times, freqs = process_audio(query_file)
            t1 = time.time()
            spec_time = (t1 - t0) * 1000
            
            t2 = time.time()
            match_name, max_score, offsets, all_scores = match_query(q_hashes, st.session_state.database)
            t3 = time.time()
            lookup_time = (t3 - t2) * 1000
            
            total_time = (time.time() - start_total) * 1000
            
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("SPECTROGRAM", f"{int(spec_time)} ms")
            m2.metric("CONSTELLATION", f"{len(times)} peaks")
            m3.metric("HASHING", f"{len(q_hashes)} hashes")
            m4.metric("DB LOOKUP", f"{int(lookup_time)} ms")
            
            st.divider()
            
            if match_name != "None":
                # Big Match Banner
                st.success("MATCH FOUND")
                st.markdown(f"<h1 style='text-align: left; color: white;'>{match_name}</h1>", unsafe_allow_html=True)
                st.caption(f"Cluster score: **{max_score}**")
                
                # Candidate Scores Bar Chart
                st.write("### CANDIDATE SCORES")
                score_df = pd.DataFrame(list(all_scores.items()), columns=['Track', 'Score']).set_index('Track')
                st.bar_chart(score_df)
                
                # Step 1: Spectrogram to Constellation
                st.write("### STEP 1 • FEATURE EXTRACTION")
                st.write("**From spectrogram to constellation**")
                c1, c2 = st.columns(2)
                with c1:
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz', cmap='magma', ax=ax1)
                    st.pyplot(fig1)
                with c2:
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    t_bins = librosa.frames_to_time(times, sr=sr, hop_length=512)
                    f_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)[list(freqs)]
                    ax2.scatter(t_bins, f_bins, color='cyan', s=5, alpha=0.7)
                    ax2.set_xlabel('time (s)')
                    ax2.set_ylabel('frequency (Hz)')
                    st.pyplot(fig2)
                
                # Step 3: The Proof
                st.write("### STEP 3 • THE PROOF")
                st.write("**The alignment spike**")
                st.write(f"A genuine match makes them converge: **{max_score} hashes agreed on a single offset.**")
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                ax3.hist(offsets, bins=100, color='orange', edgecolor='black')
                ax3.set_xlabel("time offset (database frame - query frame)")
                ax3.set_ylabel("# hashes")
                st.pyplot(fig3)
                
            else:
                st.error("No definitive match found in the database.")

# ==========================================
# TAB 3: BATCH
# ==========================================
with tab3:
    st.subheader("Identify many clips at once")
    st.markdown("Upload a set of query clips. Each is identified against the currently indexed library, and the results are written to a standardised `results.csv`.")
    
    batch_files = st.file_uploader("Upload query clips", type=["mp3", "wav"], accept_multiple_files=True, key="batch_upload")
    
    if batch_files and st.session_state.database:
        if st.button("Run batch", type="primary"):
            with st.spinner("Processing batch..."):
                results = []
                for file in batch_files:
                    q_hashes, _, _, _, _ = process_audio(file)
                    match_name, _, _, _ = match_query(q_hashes, st.session_state.database)
                    filename_no_ext = os.path.splitext(file.name)[0]
                    
                    # Convert "None" string to lowercase "none" per rubric
                    if match_name == "None":
                        match_name = "none"
                        
                    results.append({"filename": filename_no_ext, "prediction": match_name})
                
                st.write("### RESULTS")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇ Download results.csv",
                    data=csv,
                    file_name='results.csv',
                    mime='text/csv',
                )
# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# 
# import streamlit as st
# import numpy as np
# import librosa
# # ... [rest of your app code below] ...



