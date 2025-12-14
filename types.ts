export interface AnalysisResult {
  timestamp: number; // Seconds from start
  fakeProbability: number; // 0-100
  confidence: string;
  reasoning: string;
}

export enum AnalysisStatus {
  IDLE = 'IDLE',
  LOADING_AUDIO = 'LOADING_AUDIO',
  ANALYZING = 'ANALYZING',
  COMPLETED = 'COMPLETED',
  ERROR = 'ERROR',
}

export interface AudioMetadata {
  name: string;
  duration: number;
  sampleRate: number;
}