import { GoogleGenAI, Type } from "@google/genai";
import { AnalysisResult } from "../types";

// Initialize Gemini Client
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const SYSTEM_PROMPT = `
You are a world-class Forensic Audio Analyst specializing in Deepfake Detection. 
Your task is to analyze short audio segments and detect artifacts of synthetic generation.
Look for:
1. Unnatural breathing patterns or lack of breathing.
2. Metallic or robotic artifacts in high frequencies.
3. Inconsistent background noise or unnatural silence between words.
4. Pitch flattening or strange prosody.

Return a JSON object with:
- "fakeProbability": an integer between 0 and 100 (100 being certainly fake).
- "confidence": "Low", "Medium", or "High".
- "reasoning": A concise (max 15 words) explanation of what you heard.
`;

const responseSchema = {
  type: Type.OBJECT,
  properties: {
    fakeProbability: { type: Type.INTEGER },
    confidence: { type: Type.STRING },
    reasoning: { type: Type.STRING },
  },
  required: ["fakeProbability", "confidence", "reasoning"],
};

export const analyzeAudioSegment = async (
  base64Audio: string,
  timestamp: number
): Promise<AnalysisResult> => {
  try {
    const modelId = "gemini-2.5-flash"; // Fast and capable for this task

    const response = await ai.models.generateContent({
      model: modelId,
      contents: [
        {
          role: "user",
          parts: [
            {
              inlineData: {
                mimeType: "audio/wav",
                data: base64Audio,
              },
            },
            {
              text: "Analyze this audio segment for deepfake signatures.",
            },
          ],
        },
      ],
      config: {
        systemInstruction: SYSTEM_PROMPT,
        responseMimeType: "application/json",
        responseSchema: responseSchema,
        temperature: 0.2, // Low temperature for consistent detection
      },
    });

    const text = response.text;
    if (!text) throw new Error("No response from Gemini");

    const json = JSON.parse(text);

    return {
      timestamp,
      fakeProbability: json.fakeProbability,
      confidence: json.confidence,
      reasoning: json.reasoning,
    };
  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    // Fallback error result
    return {
      timestamp,
      fakeProbability: 0,
      confidence: "Error",
      reasoning: "Analysis failed due to API error.",
    };
  }
};