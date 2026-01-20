import { AnalysisResult } from "../types";

const API_URL = "http://localhost:8000/api/analyze";

export const analyzeAudioSegment = async (
  base64Audio: string,
  timestamp: number
): Promise<AnalysisResult> => {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        audio_base64: base64Audio,
        timestamp: timestamp,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API Error: ${response.status} ${errorText}`);
    }

    const json = await response.json();

    return {
      timestamp: json.timestamp,
      fakeProbability: json.fakeProbability,
      confidence: json.confidence,
      reasoning: json.reasoning,
    };
  } catch (error) {
    console.error("Local API Analysis Error:", error);
    // Fallback error result
    return {
      timestamp,
      fakeProbability: 0,
      confidence: "Error",
      reasoning: "Failed to connect to local detection server.",
    };
  }
};