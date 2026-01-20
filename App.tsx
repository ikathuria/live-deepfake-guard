import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ShieldAlert, ShieldCheck, Activity, FileAudio, Play, Pause, Upload, Cpu } from 'lucide-react';
import { MetricsCard } from './components/MetricsCard';
import { DetectionChart } from './components/DetectionChart';
import { AnalysisResult, AnalysisStatus } from './types';
import { audioBufferToWav, blobToBase64 } from './utils/audioUtils';
import { analyzeAudioSegment } from './services/geminiService'; // Keep for type signature if needed, or remove
import { loadModel, classifyAudio } from './services/inferenceService';

const CHUNK_DURATION = 2.0; // Seconds
const ANALYSIS_INTERVAL = 1500; // ms (delay between calls to simulate streaming/rate limit)

export default function App() {
  const [status, setStatus] = useState<AnalysisStatus>(AnalysisStatus.IDLE);
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [loadingProgress, setLoadingProgress] = useState<number>(0);
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [overallRisk, setOverallRisk] = useState<number>(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const startTimeRef = useRef<number>(0);
  const analysisTimerRef = useRef<number | null>(null);

  // Initialize AudioContext
  useEffect(() => {
    audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();

    // Load Model on Start
    const initModel = async () => {
      try {
        await loadModel((data: any) => {
          if (data.status === 'progress') {
            setLoadingProgress(data.progress || 0);
          }
        });
        setModelLoaded(true);
        console.log("Model Loaded");
      } catch (e) {
        console.error("Failed to load model", e);
        setStatus(AnalysisStatus.ERROR);
      }
    };
    initModel();

    return () => {
      audioContextRef.current?.close();
    };
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setStatus(AnalysisStatus.LOADING_AUDIO);
    setCurrentFile(file);
    setResults([]);
    setCurrentTime(0);
    setOverallRisk(0);
    setIsPlaying(false);

    try {
      const arrayBuffer = await file.arrayBuffer();
      if (audioContextRef.current) {
        const decodedBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
        setAudioBuffer(decodedBuffer);
        setStatus(AnalysisStatus.IDLE);
      }
    } catch (e) {
      console.error("Error decoding audio", e);
      setStatus(AnalysisStatus.ERROR);
    }
  };

  const startAnalysis = useCallback(async () => {
    if (!audioBuffer || status === AnalysisStatus.ANALYZING) return;

    setStatus(AnalysisStatus.ANALYZING);
    setIsPlaying(true);

    // Start Audio Playback for user to hear
    if (audioContextRef.current) {
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      const source = audioContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContextRef.current.destination);
      source.start(0, currentTime);
      sourceNodeRef.current = source;
      startTimeRef.current = audioContextRef.current.currentTime - currentTime;

      // Update UI timer
      const timer = window.setInterval(() => {
        if (audioContextRef.current) {
          const t = audioContextRef.current.currentTime - startTimeRef.current;
          if (t >= audioBuffer.duration) {
            stopAnalysis();
          } else {
            setCurrentTime(t);
          }
        }
      }, 100);

      (window as any).playbackInterval = timer;
    }

    // "Streaming" Analysis Loop
    let cursor = 0;
    const processNextChunk = async () => {
      if (cursor >= audioBuffer.duration) {
        setStatus(AnalysisStatus.COMPLETED);
        setIsPlaying(false);
        return;
      }

      // 1. Slice Buffer - Get raw audio data for transformers.js
      // We need Float32Array for the model, conveniently AudioBuffer gives us that.
      const startSample = Math.floor(cursor * audioBuffer.sampleRate);
      const endSample = Math.floor((cursor + CHUNK_DURATION) * audioBuffer.sampleRate);
      const channelData = audioBuffer.getChannelData(0).slice(startSample, endSample);

      // 2. Inference
      try {
        const resultData = await classifyAudio(channelData, audioBuffer.sampleRate);

        const result: AnalysisResult = {
          timestamp: cursor,
          fakeProbability: Math.round(resultData.fakeProbability),
          confidence: resultData.fakeProbability > 80 ? 'High' : 'Medium',
          reasoning: `Score: ${resultData.fakeProbability.toFixed(1)}% (Top Class: ${resultData.details[0]?.label || 'N/A'})`
        };

        // 3. Update State
        setResults(prev => {
          const newResults = [...prev, result];
          const avg = newResults.reduce((acc, curr) => acc + curr.fakeProbability, 0) / newResults.length;
          setOverallRisk(Math.round(avg));
          return newResults;
        });
      } catch (err) {
        console.error("Inference Error", err);
      }

      cursor += CHUNK_DURATION;

      // 4. Schedule next chunk (if still playing)
      if (status !== AnalysisStatus.ERROR) {
        analysisTimerRef.current = window.setTimeout(processNextChunk, ANALYSIS_INTERVAL);
      }
    };

    processNextChunk();

  }, [audioBuffer, status, currentTime]);

  const stopAnalysis = () => {
    setIsPlaying(false);
    if (status === AnalysisStatus.ANALYZING) setStatus(AnalysisStatus.COMPLETED);

    if (sourceNodeRef.current) {
      sourceNodeRef.current.stop();
      sourceNodeRef.current = null;
    }

    if ((window as any).playbackInterval) clearInterval((window as any).playbackInterval);
    if (analysisTimerRef.current) clearTimeout(analysisTimerRef.current);
  };

  const togglePlayback = () => {
    if (isPlaying) {
      stopAnalysis();
    } else {
      startAnalysis();
    }
  };

  // Safe reset when unmounting or changing file
  useEffect(() => {
    return () => stopAnalysis();
  }, [currentFile]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans p-6 md:p-10 flex flex-col gap-6">

      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-indigo-600 rounded-lg shadow-lg shadow-indigo-900/50">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">DeepGuard <span className="text-indigo-500">Analysis</span></h1>
            <p className="text-slate-400 text-sm">Real-time Forensic Deepfake Detection Pipeline</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {status === AnalysisStatus.ANALYZING && (
            <span className="flex items-center gap-2 px-3 py-1 bg-red-500/10 border border-red-500/20 text-red-500 rounded-full text-xs font-mono animate-pulse">
              <div className="w-2 h-2 rounded-full bg-red-500"></div>
              LIVE INFERENCE
            </span>
          )}
          <div className="text-right">
            <div className="text-xs text-slate-500 uppercase tracking-widest font-semibold">Model</div>
            <div className="text-indigo-400 font-mono text-sm">
              {modelLoaded ? "Wav2Vec2 (Client-Side)" : `Loading Model... ${Math.round(loadingProgress)}%`}
            </div>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* Left Col: Controls & Metrics */}
        <div className="lg:col-span-4 flex flex-col gap-6">

          {/* Control Panel */}
          <div className="bg-slate-900 rounded-xl p-6 border border-slate-800 shadow-xl">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-slate-400" />
              Input Source
            </h2>

            <div className="space-y-4">
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-700 border-dashed rounded-lg cursor-pointer bg-slate-800/50 hover:bg-slate-800 transition-colors group">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-8 h-8 mb-3 text-slate-500 group-hover:text-indigo-400 transition-colors" />
                  <p className="mb-2 text-sm text-slate-400"><span className="font-semibold">Click to upload audio</span></p>
                  <p className="text-xs text-slate-500">WAV, MP3, FLAC (Max 10MB)</p>
                </div>
                <input type="file" className="hidden" accept="audio/*" onChange={handleFileUpload} />
              </label>

              {currentFile && (
                <div className="bg-slate-800 rounded p-3 flex items-center justify-between border border-slate-700">
                  <div className="flex items-center gap-3 overflow-hidden">
                    <FileAudio className="w-8 h-8 text-indigo-400 flex-shrink-0" />
                    <div className="truncate">
                      <p className="text-sm font-medium text-white truncate">{currentFile.name}</p>
                      <p className="text-xs text-slate-500">{audioBuffer ? `${audioBuffer.duration.toFixed(1)}s • ${audioBuffer.sampleRate}Hz` : 'Decoding...'}</p>
                    </div>
                  </div>
                </div>
              )}

              <button
                onClick={togglePlayback}
                disabled={!audioBuffer || !modelLoaded}
                className={`w-full py-3 px-4 rounded-lg font-medium flex items-center justify-center gap-2 transition-all ${(!audioBuffer || !modelLoaded)
                    ? 'bg-slate-800 text-slate-600 cursor-not-allowed'
                    : isPlaying
                      ? 'bg-red-500/10 text-red-500 border border-red-500/50 hover:bg-red-500/20'
                      : 'bg-indigo-600 text-white hover:bg-indigo-500 shadow-lg shadow-indigo-900/40'
                  }`}
              >
                {isPlaying ? <><Pause className="w-4 h-4" /> Stop Analysis</> : <><Play className="w-4 h-4" /> Start Pipeline</>}
              </button>
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-4">
            <MetricsCard
              title="Risk Level"
              value={`${overallRisk}%`}
              subtext="Avg Fake Probability"
              colorClass={overallRisk > 70 ? "text-red-500" : overallRisk > 30 ? "text-yellow-500" : "text-emerald-500"}
              icon={overallRisk > 50 ? <ShieldAlert /> : <ShieldCheck />}
            />
            <MetricsCard
              title="Chunks Processed"
              value={results.length.toString()}
              subtext={`Time: ${currentTime.toFixed(1)}s`}
              colorClass="text-indigo-400"
              icon={<Activity />}
            />
          </div>

          {/* Recent Logs */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 flex-1 flex flex-col min-h-[200px] overflow-hidden">
            <div className="p-4 border-b border-slate-800 bg-slate-900/50">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Analysis Log</h3>
            </div>
            <div className="overflow-y-auto flex-1 p-0 scroll-smooth">
              {results.length === 0 ? (
                <div className="h-full flex items-center justify-center text-slate-600 text-sm italic p-4">
                  Waiting for pipeline execution...
                </div>
              ) : (
                <table className="w-full text-left text-sm">
                  <thead className="bg-slate-950 text-slate-500 font-medium text-xs uppercase">
                    <tr>
                      <th className="px-4 py-2">Time</th>
                      <th className="px-4 py-2">Prob</th>
                      <th className="px-4 py-2">Reason</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {[...results].reverse().map((r, i) => (
                      <tr key={i} className="hover:bg-slate-800/50 transition-colors">
                        <td className="px-4 py-2 font-mono text-slate-400">{r.timestamp.toFixed(1)}s</td>
                        <td className={`px-4 py-2 font-bold ${r.fakeProbability > 50 ? 'text-red-400' : 'text-emerald-400'}`}>
                          {r.fakeProbability}%
                        </td>
                        <td className="px-4 py-2 text-slate-300 text-xs truncate max-w-[150px]" title={r.reasoning}>
                          {r.reasoning}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>

        {/* Right Col: Visualization */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          <div className="bg-slate-900 rounded-xl p-1 border border-slate-800 shadow-xl h-[400px] relative group">
            <div className="absolute top-4 left-4 z-10 bg-slate-950/80 backdrop-blur px-3 py-1 rounded border border-slate-700">
              <span className="text-xs font-semibold text-slate-300">Live Probability Stream</span>
            </div>
            {results.length > 0 ? (
              <DetectionChart data={results} />
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center text-slate-600">
                <Activity className="w-16 h-16 mb-4 opacity-20" />
                <p>No data to visualize yet.</p>
                <p className="text-sm">Upload audio and start pipeline.</p>
              </div>
            )}
          </div>

          {/* Simulated Spectrogram Area (Visual Placeholder) */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 flex-1 relative overflow-hidden">
            <h3 className="text-sm font-semibold text-slate-400 mb-4 uppercase tracking-widest">Active Waveform Segment</h3>
            <div className="flex items-center justify-center h-32 gap-1">
              {Array.from({ length: 60 }).map((_, i) => {
                // Simulate visualization bars
                const isActive = isPlaying;
                const height = isActive ? Math.random() * 100 : 20;
                return (
                  <div
                    key={i}
                    className={`w-2 rounded-full transition-all duration-100 ${isActive ? 'bg-indigo-500' : 'bg-slate-700'}`}
                    style={{ height: `${height}%` }}
                  />
                )
              })}
            </div>
            {/* Progress Line */}
            <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]"></div>
            <div className="absolute bottom-2 right-4 text-xs font-mono text-slate-500">
              Processing Window: {CHUNK_DURATION}s
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}