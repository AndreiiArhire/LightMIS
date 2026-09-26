# LightMIS mobile benchmark

Android LiteRT 1.4.1 runner and ADB controller for `.tflite` models with one FLOAT32 input of shape `[1, 3, 256, 256]`. `--inference-mode FP16` allows reduced GPU precision; it does not convert model weights. Network latency measures `Interpreter.run`, excluding initialization, warm-up and the requested delay.

## Run

Requires Java 17, Android SDK 35, Python 3 and one authorized device in `adb devices`. Place the `.tflite` models directly in `--models-dir`. From this directory:

```bash
bash ./gradlew :app:assembleDebug
adb devices

python3 tools/run_mobile_suite.py \
  --apk "$PWD/app/build/outputs/apk/debug/app-debug.apk" \
  --models-dir "/path/to/litert/single_model/LightMIS" \
  --output-dir "/path/to/litert/results/LightMIS/experiment_fp16_t4_d100" \
  --mode latency \
  --inference-mode FP16 \
  --cpu-threads 4 \
  --inter-inference-delay-ms 100 \
  --sessions 5 \
  --warmup-runs 20 \
  --measured-runs 50 \
  --max-battery-temp-c 40 \
  --temperature-poll-seconds 10 \
  --timeout-seconds 1800 \
  --random-seed 12345

python3 tools/summarize_mobile_results.py \
  --results-dir "/path/to/litert/results/LightMIS/experiment_fp16_t4_d100"
  
  
python3 tools/run_mobile_suite.py \
  --apk "$PWD/app/build/outputs/apk/debug/app-debug.apk" \
  --models-dir "/path/to/litert/single_model/LightMIS" \
  --output-dir "/path/to/litert/results/LightMIS/experiment_fp16_t4_d100_memory" \
  --mode memory \
  --inference-mode FP16 \
  --cpu-threads 4 \
  --inter-inference-delay-ms 100 \
  --memory-interval-ms 5 \
  --sessions 5 \
  --warmup-runs 20 \
  --measured-runs 50 \
  --max-battery-temp-c 40 \
  --temperature-poll-seconds 10 \
  --timeout-seconds 1800 \
  --random-seed 12345 \
  --skip-install
  
python3 tools/summarize_mobile_results.py \
  --results-dir "/path/to/litert/results/LightMIS/experiment_fp16_t4_d100_memory"
```

Each output directory must be new or empty. The first run installs the APK; the memory run reuses it with --skip-install. Report latency from the latency run and maximum sampled process RSS from the memory run; do not report the memory run’s instrumented latency. Each run records 250 measured invocations per model across five sessions. The 40 °C option is a start threshold, not the observed temperature.

