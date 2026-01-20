
$baseUrl = "https://huggingface.co/Xenova/wav2vec2-base-superb-ks/resolve/main/"
$outputDir = "public\models\Xenova\wav2vec2-base-superb-ks"
$files = @("config.json", "preprocessor_config.json", "onnx/model_quantized.onnx")

New-Item -ItemType Directory -Force -Path $outputDir
New-Item -ItemType Directory -Force -Path "$outputDir\onnx"

foreach ($file in $files) {
    $url = $baseUrl + $file
    $output = Join-Path $outputDir $file
    Write-Host "Downloading $file from $url to $output"
    try {
        Invoke-WebRequest -Uri $url -OutFile $output
        Write-Host "Success."
    } catch {
        Write-Error "Failed to download $file : $_"
    }
}
