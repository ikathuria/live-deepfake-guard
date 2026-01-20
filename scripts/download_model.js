
import fs from 'node:fs';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';
import { createWriteStream } from 'node:fs';

const MODEL_ID = 'Xenova/wav2vec2-base-superb-ks';
const FILES = [
	'config.json',
	'preprocessor_config.json',
	'onnx/model_quantized.onnx'
];

const BASE_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/`;
const OUTPUT_DIR = path.join('public', 'models', MODEL_ID);

async function downloadFile(filename) {
	const url = BASE_URL + filename;
	const outputPath = path.join(OUTPUT_DIR, filename);
	const dir = path.dirname(outputPath);

	if (!fs.existsSync(dir)) {
		fs.mkdirSync(dir, { recursive: true });
	}

	console.log(`Downloading ${filename} from ${url}...`);

	try {
		const response = await fetch(url);
		if (!response.ok) {
			throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
		}

		await pipeline(response.body, createWriteStream(outputPath));
		console.log(`Saved ${filename}`);
	} catch (error) {
		console.error(`Error downloading ${filename}:`, error);
		throw error;
	}
}

async function main() {
	console.log(`Starting download for ${MODEL_ID}...`);
	for (const file of FILES) {
		await downloadFile(file);
	}
	console.log('All downloads finished.');
}

main();
