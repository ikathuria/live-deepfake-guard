import streamlit as st
import time
import numpy as np
import pandas as pd
import altair as alt
from stream_simulator import generate_audio_stream
from inference_engine import DeepfakeDetector
import os

# Page Config
st.set_page_config(page_title="Deepfake Voice Guard", page_icon="🛡️", layout="wide")

# Title and Description
st.title("🛡️ Live Deepfake Voice Detection")
st.markdown("Real-time monitoring of audio stream for synthetic voice signatures.")


# Initialize components
@st.cache_resource
def load_detector():
    return DeepfakeDetector()


if "detector" not in st.session_state:
    with st.spinner("Loading Model... (This may take a moment)"):
        st.session_state.detector = load_detector()
        st.success("Model Loaded!")

# Sidebar controls
st.sidebar.header("Configuration")
chunk_duration = st.sidebar.slider("Chunk Duration (s)", 1, 5, 2)
threshold = st.sidebar.slider("Alert Threshold", 0.0, 1.0, 0.8)
audio_file = st.sidebar.text_input("Audio Source File", "sample_audio.wav")

# State for stopping
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("Start Detection", type="primary")
with col2:
    stop_btn = st.button("Stop")

if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False


# Main Dashboard Area
placeholder = st.empty()
alert_box = st.empty()

# Data buffer for plotting
if "history" not in st.session_state:
    st.session_state.history = []


def run_detection():
    # Verify file exists or use dummy if empty string
    file_to_use = audio_file if audio_file and os.path.exists(audio_file) else None

    if not file_to_use:
        st.warning(f"File '{audio_file}' not found. Using simulated noise/silence.")
        # Create a dummy file or just pass a path that the generator handles gracefully (generator handles non-existent)
        file_to_use = "dummy.wav"

    stream = generate_audio_stream(file_to_use, chunk_duration=chunk_duration)

    # Create chart container
    chart_placeholder = st.empty()

    for audio_chunk in stream:
        if not st.session_state.running:
            break

        # Inference
        prob = st.session_state.detector.predict(audio_chunk)

        # Update History
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.history.append({"Time": timestamp, "Fake Probability": prob})
        if len(st.session_state.history) > 20:
            st.session_state.history.pop(0)

        df = pd.DataFrame(st.session_state.history)

        # Visualization
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x="Time",
                y=alt.Y("Fake Probability", scale=alt.Scale(domain=[0, 1])),
                color=alt.value("#ff4b4b" if prob > threshold else "#00c0f2"),
            )
            .properties(height=300)
        )

        chart_placeholder.altair_chart(chart, use_container_width=True)

        # Alert Logic
        if prob > threshold:
            alert_box.error(
                f"🚨 ALERT: High Probability of Deepfake Detected! ({prob:.2%})"
            )
        else:
            alert_box.success(f"Status: Secure ({prob:.2%})")

        # UI refresh rate control implies we process as fast as chunks come or sleep?
        # The generator yields immediately for file reading, so we add a small sleep to simulate real-time playback
        # roughly matching chunk duration if we want "real-time" feel, or just fast processing.
        # User constraint: "Latency < 200ms". This refers to processing time.
        # We will sleep a bit to make the graph readable, not instantaneous.
        time.sleep(0.1)


if st.session_state.running:
    run_detection()
