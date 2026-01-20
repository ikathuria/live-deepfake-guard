import time
import numpy as np
import librosa
import os

def generate_audio_stream(file_path, chunk_duration=2, sr=16000):
    """
    Simulates a live audio stream by yielding chunks of audio data.
    
    Args:
        file_path (str): Path to the audio file to stream.
        chunk_duration (int): Duration of each chunk in seconds.
        sr (int): Sampling rate.
        
    Yields:
        np.ndarray: Audio chunk.
    """
    if not os.path.exists(file_path):
        # If file doesn't exist, yield random noise for testing purposes ensures stability
        print(f"Warning: File {file_path} not found. Streaming silence/noise.")
        while True:
            yield np.random.uniform(-0.01, 0.01, size=(int(chunk_duration * sr),))
            time.sleep(chunk_duration)

    try:
        # Load the full audio file
        # Note: For very large files, this should be done with soundfile streaming.
        # Given constraints, standard librosa load is acceptable if file fits in memory.
        audio, _ = librosa.load(file_path, sr=sr)
        chunk_samples = int(chunk_duration * sr)
        total_samples = len(audio)
        
        while True: # Loop indefinitely to simulate continuous stream
            for start in range(0, total_samples, chunk_samples):
                end = min(start + chunk_samples, total_samples)
                chunk = audio[start:end]
                
                # Pad if last chunk is smaller than expected
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
                    
                yield chunk
                # Simulate real-time latency (optional, but good for "simulation")
                # time.sleep(chunk_duration) # Logic usually controlled by consumer rate, but here we just yield.
                
    except Exception as e:
        print(f"Error in stream generator: {e}")
        # Fallback to silence to prevent crash
        while True:
             yield np.zeros(int(chunk_duration * sr))
