// Utility to convert AudioBuffer to WAV format (Blob)
// Gemini prefers a recognized container like WAV over raw PCM for simple inlineData calls
export const audioBufferToWav = (buffer: AudioBuffer, start: number, duration: number): Blob => {
  const sampleRate = buffer.sampleRate;
  const startOffset = Math.floor(start * sampleRate);
  const endOffset = Math.floor((start + duration) * sampleRate);
  
  // Handle edge case where we are at the end of the file
  const actualEndOffset = Math.min(endOffset, buffer.length);
  const frameCount = actualEndOffset - startOffset;
  
  if (frameCount <= 0) {
    return new Blob([]);
  }

  const numChannels = 1; // Downmix to mono for analysis efficiency
  const length = frameCount * numChannels * 2; // 16-bit
  const arrayBuffer = new ArrayBuffer(44 + length);
  const view = new DataView(arrayBuffer);

  // RIFF identifier
  writeString(view, 0, 'RIFF');
  // file length
  view.setUint32(4, 36 + length, true);
  // RIFF type
  writeString(view, 8, 'WAVE');
  // format chunk identifier
  writeString(view, 12, 'fmt ');
  // format chunk length
  view.setUint32(16, 16, true);
  // sample format (raw)
  view.setUint16(20, 1, true);
  // channel count
  view.setUint16(22, numChannels, true);
  // sample rate
  view.setUint32(24, sampleRate, true);
  // byte rate (sample rate * block align)
  view.setUint32(28, sampleRate * 2, true);
  // block align (channel count * bytes per sample)
  view.setUint16(32, 2, true);
  // bits per sample
  view.setUint16(34, 16, true);
  // data chunk identifier
  writeString(view, 36, 'data');
  // data chunk length
  view.setUint32(40, length, true);

  // Write PCM samples
  const channelData = buffer.getChannelData(0); // Use first channel (mono)
  let offset = 44;
  for (let i = 0; i < frameCount; i++) {
    const originalIndex = startOffset + i;
    // Clamp sample to -1 to 1
    let sample = Math.max(-1, Math.min(1, channelData[originalIndex]));
    // Scale to 16-bit integer
    sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    view.setInt16(offset, sample, true);
    offset += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
};

const writeString = (view: DataView, offset: number, string: string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};

export const blobToBase64 = (blob: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      // Remove data URL prefix (e.g., "data:audio/wav;base64,")
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};