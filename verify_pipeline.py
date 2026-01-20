import time
import numpy as np
import scipy.io.wavfile as wavfile
from stream_simulator import generate_audio_stream
from inference_engine import DeepfakeDetector
import os

def test_pipeline():
    print("1. Creating dummy audio file...")
    dummy_audio = np.random.uniform(-0.5, 0.5, size=(16000 * 10,)) # 10 seconds
    # scipy expects int16 or float32. standard is int16 for wav usually to avoid issues, but float is fine.
    wavfile.write('sample_audio.wav', 16000, dummy_audio.astype(np.float32))
    print("   Done.")

    print("\n2. Initializing Detector...")
    try:
        detector = DeepfakeDetector()
        print("   Detector initialized.")
    except Exception as e:
        print(f"   FAILED to initialize detector: {e}")
        return

    print("\n3. Testing Stream Generator...")
    stream = generate_audio_stream('sample_audio.wav', chunk_duration=1) # 1s chunks
    
    print("\n4. Running Inference Loop (5 chunks)...")
    latencies = []
    
    try:
        count = 0
        for chunk in stream:
            count += 1
            if count > 5: break
            
            t0 = time.time()
            prob = detector.predict(chunk)
            t1 = time.time()
            
            latency = (t1 - t0) * 1000
            latencies.append(latency)
            
            print(f"   Chunk {count}: Prob={prob:.4f}, Latency={latency:.2f}ms")
            
            # Constraints check
            if latency > 200:
                print(f"   WARNING: Latency exceeded 200ms limit!")
            if not (0.0 <= prob <= 1.0):
                 print(f"   ERROR: Probability out of bounds!")

    except Exception as e:
        print(f"   CRASHED during loop: {e}")
        return

    avg_latency = np.mean(latencies)
    print(f"\nAverage Latency: {avg_latency:.2f}ms")
    if avg_latency < 200:
        print("PASS: Latency constraint met.")
    else:
        print("FAIL: Latency constraint exceeded.")

    print("\nVerification Complete.")

if __name__ == "__main__":
    test_pipeline()
