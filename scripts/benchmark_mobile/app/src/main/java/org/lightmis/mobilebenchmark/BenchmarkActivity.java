package org.lightmis.mobilebenchmark;

import android.app.Activity;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
import android.os.SystemClock;
import android.util.Log;
import android.view.WindowManager;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Headless-by-ADB LiteRT benchmark Activity.
 *
 * Every invocation creates a fresh Interpreter and GPU delegate. The host
 * controller force-stops the process between sessions. GPU delegate creation
 * and inference happen on this same worker thread, as required by LiteRT.
 */
public final class BenchmarkActivity extends Activity {
    private static final String TAG = "LightMISBench";
    private static final String LITERT_VERSION = "1.4.1";
    private static final int[] REQUIRED_INPUT_SHAPE = {1, 3, 256, 256};

    private TextView statusView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        statusView = new TextView(this);
        statusView.setText("Benchmark running...");
        statusView.setPadding(32, 32, 32, 32);
        setContentView(statusView);

        Thread worker = new Thread(this::runBenchmark, "litert-benchmark-worker");
        worker.start();
    }

    private void runBenchmark() {
        long activityStartNs = SystemClock.elapsedRealtimeNanos();
        String error = null;
        JSONObject result = new JSONObject();
        GpuDelegate gpuDelegate = null;
        Interpreter interpreter = null;
        MemorySampler memorySampler = null;

        try {
            Intent intent = getIntent();
            String modelName = requireExtra(intent, "model_name");
            String relativeModelPath = requireExtra(intent, "model_path");
            String resultName = requireExtra(intent, "result_name");
            String mode = intent.getStringExtra("mode");
            if (mode == null) mode = "latency";
            int session = intent.getIntExtra("session", 1);
            int warmupRuns = intent.getIntExtra("warmup_runs", 20);
            int measuredRuns = intent.getIntExtra("measured_runs", 100);
            int interInferenceDelayMs = intent.getIntExtra("inter_inference_delay_ms", 0);
            int cpuThreads = intent.getIntExtra("cpu_threads", 1);
            String inferenceMode = intent.getStringExtra("inference_mode");
            if (inferenceMode == null) inferenceMode = "FP16";
            if (interInferenceDelayMs < 0 || cpuThreads < 1
                    || warmupRuns < 0 || measuredRuns < 1) {
                throw new IllegalArgumentException("Invalid delay, threads or run counts");
            }
            if (!"FP16".equals(inferenceMode) && !"FP32".equals(inferenceMode)) {
                throw new IllegalArgumentException("inference_mode must be FP16 or FP32");
            }
            boolean precisionLossAllowed = "FP16".equals(inferenceMode);
            int memoryIntervalMs = intent.getIntExtra("memory_interval_ms", 50);
            String relativeImagePath = intent.getStringExtra("image_path");
            String normalization = intent.getStringExtra("normalization");
            if (normalization == null) normalization = "zscore";

            validateSafeRelativePath(relativeModelPath);
            validateSafeRelativePath(resultName);
            if (relativeImagePath != null && !relativeImagePath.isEmpty()) {
                validateSafeRelativePath(relativeImagePath);
            }

            File modelFile = new File(getFilesDir(), relativeModelPath);
            File resultFile = new File(new File(getFilesDir(), "results"), resultName);
            if (!modelFile.isFile()) {
                throw new IOException("Model does not exist: " + modelFile);
            }
            if (!resultFile.getParentFile().exists() && !resultFile.getParentFile().mkdirs()) {
                throw new IOException("Cannot create result directory");
            }

            double batteryTempStartC = getBatteryTemperatureC();
            int thermalStatusStart = getThermalStatus();
            boolean powerSaveStart = isPowerSaveMode();

            long initStartNs = SystemClock.elapsedRealtimeNanos();
            MappedByteBuffer modelBuffer = mapFile(modelFile);
            GpuDelegate.Options gpuOptions = new GpuDelegate.Options();
            gpuOptions.setPrecisionLossAllowed(precisionLossAllowed);
            gpuDelegate = new GpuDelegate(gpuOptions);
            Interpreter.Options interpreterOptions = new Interpreter.Options();
            interpreterOptions.setNumThreads(cpuThreads);
            interpreterOptions.addDelegate(gpuDelegate);
            interpreter = new Interpreter(modelBuffer, interpreterOptions);
            interpreter.allocateTensors();
            long initEndNs = SystemClock.elapsedRealtimeNanos();

            Tensor inputTensor = interpreter.getInputTensor(0);
            Tensor outputTensor = interpreter.getOutputTensor(0);
            int[] inputShape = inputTensor.shape();
            if (!Arrays.equals(inputShape, REQUIRED_INPUT_SHAPE)) {
                throw new IllegalArgumentException(
                        "Expected input shape " + Arrays.toString(REQUIRED_INPUT_SHAPE)
                                + " but model has " + Arrays.toString(inputShape));
            }
            if (inputTensor.dataType() != DataType.FLOAT32) {
                throw new IllegalArgumentException(
                        "This runner currently requires FLOAT32 input, found "
                                + inputTensor.dataType());
            }

            ByteBuffer networkInput = ByteBuffer.allocateDirect(inputTensor.numBytes())
                    .order(ByteOrder.nativeOrder());
            ByteBuffer networkOutput = ByteBuffer.allocateDirect(outputTensor.numBytes())
                    .order(ByteOrder.nativeOrder());
            fillDeterministicInput(networkInput);

            PipelineProcessor pipelineProcessor = null;
            if ("pipeline".equals(mode)) {
                if (relativeImagePath == null || relativeImagePath.isEmpty()) {
                    throw new IllegalArgumentException("pipeline mode requires image_path");
                }
                File imageFile = new File(getFilesDir(), relativeImagePath);
                Bitmap pipelineBitmap = BitmapFactory.decodeFile(imageFile.getAbsolutePath());
                if (pipelineBitmap == null) {
                    throw new IOException("Cannot decode image: " + imageFile);
                }
                pipelineProcessor = new PipelineProcessor(pipelineBitmap, outputTensor);
            }

            if ("memory".equals(mode)) {
                memorySampler = new MemorySampler(memoryIntervalMs);
                memorySampler.start();
            }

            Log.i(TAG, String.format(Locale.US,
                    "BEGIN model=%s session=%d mode=%s warmup=%d measured=%d",
                    modelName, session, mode, warmupRuns, measuredRuns));

            // Warm-up runs are never included in latency statistics.
            for (int i = 0; i < warmupRuns; i++) {
                if (pipelineProcessor != null) {
                    pipelineProcessor.prepareInput(networkInput, normalization);
                    networkOutput.rewind();
                    interpreter.run(networkInput, networkOutput);
                    pipelineProcessor.foregroundPixelCount(networkOutput);
                } else {
                    networkInput.rewind();
                    networkOutput.rewind();
                    interpreter.run(networkInput, networkOutput);
                }
                // Includes the boundary from last warm-up to first measured inference.
                if (interInferenceDelayMs > 0) {
                    SystemClock.sleep(interInferenceDelayMs);
                }
            }

            List<Double> networkTimesMs = new ArrayList<>(measuredRuns);
            List<Double> pipelineTimesMs = new ArrayList<>(measuredRuns);
            long maskChecksum = 0L;

            for (int i = 0; i < measuredRuns; i++) {
                if ("pipeline".equals(mode)) {
                    long pipelineStartNs = SystemClock.elapsedRealtimeNanos();
                    pipelineProcessor.prepareInput(networkInput, normalization);
                    networkOutput.rewind();
                    long networkStartNs = SystemClock.elapsedRealtimeNanos();
                    interpreter.run(networkInput, networkOutput);
                    long networkEndNs = SystemClock.elapsedRealtimeNanos();
                    maskChecksum += pipelineProcessor.foregroundPixelCount(networkOutput);
                    long pipelineEndNs = SystemClock.elapsedRealtimeNanos();
                    networkTimesMs.add(nsToMs(networkEndNs - networkStartNs));
                    pipelineTimesMs.add(nsToMs(pipelineEndNs - pipelineStartNs));
                } else {
                    networkInput.rewind();
                    networkOutput.rewind();
                    long startNs = SystemClock.elapsedRealtimeNanos();
                    interpreter.run(networkInput, networkOutput);
                    long endNs = SystemClock.elapsedRealtimeNanos();
                    networkTimesMs.add(nsToMs(endNs - startNs));
                }
                // Outside both network and pipeline timing. No sleep after final run.
                if (interInferenceDelayMs > 0 && i < measuredRuns - 1) {
                    SystemClock.sleep(interInferenceDelayMs);
                }
            }

            if (memorySampler != null) {
                memorySampler.requestStop();
                memorySampler.join();
            }

            double batteryTempEndC = getBatteryTemperatureC();
            int thermalStatusEnd = getThermalStatus();
            boolean powerSaveEnd = isPowerSaveMode();

            result.put("schema_version", 2);
            result.put("model", modelName);
            result.put("session", session);
            result.put("mode", mode);
            result.put("litert_version", LITERT_VERSION);
            result.put("gpu_precision_loss_allowed", precisionLossAllowed);
            result.put("inference_mode", inferenceMode);
            result.put("cpu_threads", cpuThreads);
            result.put("inter_inference_delay_ms", interInferenceDelayMs);
            result.put("delay_applies_to", "warmup_and_measured; excluded_from_latency");
            result.put("precision_note", "FP16 permits reduced GPU precision; does not guarantee all ops FP16 or convert weights. CPU threads apply to supported CPU ops.");
            result.put("warmup_runs", warmupRuns);
            result.put("measured_runs", measuredRuns);
            result.put("initialization_ms", nsToMs(initEndNs - initStartNs));
            result.put("input_shape", intArrayToJson(inputShape));
            result.put("input_dtype", inputTensor.dataType().toString());
            result.put("output_shape", intArrayToJson(outputTensor.shape()));
            result.put("output_dtype", outputTensor.dataType().toString());
            result.put("model_file_bytes", modelFile.length());
            result.put("network_latency_ms", doubleListToJson(networkTimesMs));
            result.put("pipeline_latency_ms", doubleListToJson(pipelineTimesMs));
            result.put("network_summary", summarize(networkTimesMs));
            if (!pipelineTimesMs.isEmpty()) {
                result.put("pipeline_summary", summarize(pipelineTimesMs));
                result.put("pipeline_definition",
                        "in-memory 256x256 RGB bitmap + per-channel normalization + NCHW tensor copy + inference + two-class argmax; file I/O and prior dataset conversion excluded");
                result.put("normalization", normalization);
                result.put("mask_checksum", maskChecksum);
            }
            if (memorySampler != null) {
                result.put("peak_process_rss_kb", memorySampler.getPeakRssKb());
                result.put("memory_sampling_interval_ms", memoryIntervalMs);
                result.put("memory_measurement_note",
                        "Approximate whole-process VmRSS; measured in a separate instrumented run.");
            } else {
                result.put("peak_process_rss_kb", JSONObject.NULL);
            }
            result.put("battery_temperature_start_c", finiteOrNull(batteryTempStartC));
            result.put("battery_temperature_end_c", finiteOrNull(batteryTempEndC));
            result.put("thermal_status_start", thermalStatusStart);
            result.put("thermal_status_end", thermalStatusEnd);
            result.put("power_save_start", powerSaveStart);
            result.put("power_save_end", powerSaveEnd);
            result.put("device_manufacturer", Build.MANUFACTURER);
            result.put("device_model", Build.MODEL);
            result.put("android_version", Build.VERSION.RELEASE);
            result.put("android_api", Build.VERSION.SDK_INT);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                result.put("soc_manufacturer", Build.SOC_MANUFACTURER);
                result.put("soc_model", Build.SOC_MODEL);
            }
            result.put("activity_total_ms",
                    nsToMs(SystemClock.elapsedRealtimeNanos() - activityStartNs));
            result.put("error", JSONObject.NULL);

            writeJsonAtomically(resultFile, result);
            Log.i(TAG, "RESULT_FILE " + resultFile.getAbsolutePath());
            Log.i(TAG, String.format(Locale.US,
                    "END model=%s session=%d p50=%.4fms p95=%.4fms",
                    modelName, session,
                    result.getJSONObject("network_summary").getDouble("p50_ms"),
                    result.getJSONObject("network_summary").getDouble("p95_ms")));
        } catch (Throwable throwable) {
            error = Log.getStackTraceString(throwable);
            Log.e(TAG, "Benchmark failed", throwable);
            try {
                result.put("error", error);
                result.put("activity_total_ms",
                        nsToMs(SystemClock.elapsedRealtimeNanos() - activityStartNs));
                File errorDir = new File(getFilesDir(), "results");
                if (!errorDir.exists()) errorDir.mkdirs();
                writeJsonAtomically(new File(errorDir, "last_error.json"), result);
            } catch (Throwable ignored) {
                Log.e(TAG, "Could not write error result", ignored);
            }
        } finally {
            if (memorySampler != null) {
                memorySampler.requestStop();
            }
            if (interpreter != null) interpreter.close();
            if (gpuDelegate != null) gpuDelegate.close();
            final String finalError = error;
            runOnUiThread(() -> {
                statusView.setText(finalError == null ? "Benchmark complete" : "Benchmark failed; inspect logcat");
                statusView.postDelayed(this::finishAndRemoveTask, 750L);
            });
        }
    }

    private static String requireExtra(Intent intent, String name) {
        String value = intent.getStringExtra(name);
        if (value == null || value.isEmpty()) {
            throw new IllegalArgumentException("Missing Intent extra: " + name);
        }
        return value;
    }

    private static void validateSafeRelativePath(String path) {
        if (path.startsWith("/") || path.contains("..")) {
            throw new IllegalArgumentException("Unsafe relative path: " + path);
        }
    }

    private static MappedByteBuffer mapFile(File file) throws IOException {
        try (FileInputStream input = new FileInputStream(file);
             FileChannel channel = input.getChannel()) {
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
        }
    }

    private static void fillDeterministicInput(ByteBuffer buffer) {
        buffer.rewind();
        Random random = new Random(20260912L);
        while (buffer.remaining() >= Float.BYTES) {
            buffer.putFloat((float) (random.nextGaussian() * 0.25));
        }
        buffer.rewind();
    }

    private double getBatteryTemperatureC() {
        Intent battery = registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        if (battery == null) return Double.NaN;
        int tenths = battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Integer.MIN_VALUE);
        return tenths == Integer.MIN_VALUE ? Double.NaN : tenths / 10.0;
    }

    private int getThermalStatus() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return -1;
        PowerManager manager = (PowerManager) getSystemService(POWER_SERVICE);
        return manager == null ? -1 : manager.getCurrentThermalStatus();
    }

    private boolean isPowerSaveMode() {
        PowerManager manager = (PowerManager) getSystemService(POWER_SERVICE);
        return manager != null && manager.isPowerSaveMode();
    }

    private static JSONObject summarize(List<Double> values) throws Exception {
        if (values.isEmpty()) throw new IllegalArgumentException("No values to summarize");
        List<Double> sorted = new ArrayList<>(values);
        sorted.sort(Double::compareTo);
        double sum = 0.0;
        for (double value : values) sum += value;
        double mean = sum / values.size();
        double squared = 0.0;
        for (double value : values) {
            double delta = value - mean;
            squared += delta * delta;
        }
        double sampleStd = values.size() > 1
                ? Math.sqrt(squared / (values.size() - 1)) : 0.0;
        JSONObject summary = new JSONObject();
        summary.put("count", values.size());
        summary.put("mean_ms", mean);
        summary.put("sample_sd_ms", sampleStd);
        summary.put("min_ms", sorted.get(0));
        summary.put("max_ms", sorted.get(sorted.size() - 1));
        summary.put("p50_ms", percentile(sorted, 50.0));
        summary.put("p95_ms", percentile(sorted, 95.0));
        return summary;
    }

    private static double percentile(List<Double> sorted, double percentile) {
        if (sorted.size() == 1) return sorted.get(0);
        double position = (percentile / 100.0) * (sorted.size() - 1);
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);
        if (lower == upper) return sorted.get(lower);
        double weight = position - lower;
        return sorted.get(lower) * (1.0 - weight) + sorted.get(upper) * weight;
    }

    private static JSONArray doubleListToJson(List<Double> values) {
        JSONArray array = new JSONArray();
        for (double value : values) array.put(finiteOrNull(value));
        return array;
    }

    private static JSONArray intArrayToJson(int[] values) {
        JSONArray array = new JSONArray();
        for (int value : values) array.put(value);
        return array;
    }

    private static Object finiteOrNull(double value) {
        return Double.isFinite(value) ? value : JSONObject.NULL;
    }

    private static double nsToMs(long nanoseconds) {
        return nanoseconds / 1_000_000.0;
    }

    private static void writeJsonAtomically(File destination, JSONObject data) throws IOException {
        File temporary = new File(destination.getParentFile(), destination.getName() + ".tmp");
        try (FileWriter writer = new FileWriter(temporary)) {
            writer.write(data.toString());
            writer.write("\n");
        }
        if (destination.exists() && !destination.delete()) {
            throw new IOException("Cannot replace result: " + destination);
        }
        if (!temporary.renameTo(destination)) {
            throw new IOException("Cannot finalize result: " + destination);
        }
    }

    private static final class MemorySampler extends Thread {
        private final int intervalMs;
        private volatile boolean running = true;
        private volatile long peakRssKb = 0L;

        MemorySampler(int intervalMs) {
            super("rss-sampler");
            this.intervalMs = Math.max(5, intervalMs);
        }

        @Override
        public void run() {
            while (running) {
                peakRssKb = Math.max(peakRssKb, readVmRssKb());
                SystemClock.sleep(intervalMs);
            }
            peakRssKb = Math.max(peakRssKb, readVmRssKb());
        }

        void requestStop() {
            running = false;
            interrupt();
        }

        long getPeakRssKb() {
            return peakRssKb;
        }

        private static long readVmRssKb() {
            try (BufferedReader reader = new BufferedReader(new FileReader("/proc/self/status"))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.startsWith("VmRSS:")) {
                        String digits = line.replaceAll("[^0-9]", "");
                        return digits.isEmpty() ? 0L : Long.parseLong(digits);
                    }
                }
            } catch (Throwable ignored) {
                // A zero is retained when the OEM blocks the procfs field.
            }
            return 0L;
        }
    }

    private static final class PipelineProcessor {
        private static final int WIDTH = 256;
        private static final int HEIGHT = 256;
        private static final int PLANE = WIDTH * HEIGHT;
        private final Bitmap bitmap;
        private final int[] pixels = new int[PLANE];
        private final float[][] channels = new float[3][PLANE];
        private final float[] outputValues = new float[2 * PLANE];

        PipelineProcessor(Bitmap bitmap, Tensor outputTensor) {
            if (bitmap.getWidth() != WIDTH || bitmap.getHeight() != HEIGHT) {
                throw new IllegalArgumentException(
                        "Pipeline image must already be 256x256; prior dataset conversion is excluded");
            }
            if (outputTensor.dataType() != DataType.FLOAT32) {
                throw new IllegalArgumentException("Pipeline argmax requires FLOAT32 output");
            }
            int[] shape = outputTensor.shape();
            if (shape.length != 4 || shape[0] != 1 || shape[1] != 2
                    || shape[2] != HEIGHT || shape[3] != WIDTH) {
                throw new IllegalArgumentException(
                        "Pipeline argmax requires output [1,2,256,256], found "
                                + Arrays.toString(shape));
            }
            this.bitmap = bitmap;
        }

        void prepareInput(ByteBuffer destination, String normalization) {
            bitmap.getPixels(pixels, 0, WIDTH, 0, 0, WIDTH, HEIGHT);
            for (int i = 0; i < pixels.length; i++) {
                int pixel = pixels[i];
                channels[0][i] = (pixel >> 16) & 0xff;
                channels[1][i] = (pixel >> 8) & 0xff;
                channels[2][i] = pixel & 0xff;
            }

            if ("zscore".equals(normalization)) {
                for (int c = 0; c < 3; c++) {
                    double mean = 0.0;
                    for (float value : channels[c]) mean += value;
                    mean /= channels[c].length;
                    double variance = 0.0;
                    for (float value : channels[c]) {
                        double delta = value - mean;
                        variance += delta * delta;
                    }
                    double std = Math.sqrt(variance / channels[c].length);
                    if (std < 1e-8) std = 1.0;
                    for (int i = 0; i < channels[c].length; i++) {
                        channels[c][i] = (float) ((channels[c][i] - mean) / std);
                    }
                }
            } else if ("zero_one".equals(normalization)) {
                for (int c = 0; c < 3; c++) {
                    for (int i = 0; i < channels[c].length; i++) {
                        channels[c][i] /= 255.0f;
                    }
                }
            } else if (!"none".equals(normalization)) {
                throw new IllegalArgumentException("Unknown normalization: " + normalization);
            }

            destination.rewind();
            for (int c = 0; c < 3; c++) {
                for (float value : channels[c]) destination.putFloat(value);
            }
            destination.rewind();
        }

        long foregroundPixelCount(ByteBuffer output) {
            output.rewind();
            output.asFloatBuffer().get(outputValues);
            long foreground = 0L;
            for (int i = 0; i < PLANE; i++) {
                if (outputValues[PLANE + i] > outputValues[i]) foreground++;
            }
            output.rewind();
            return foreground;
        }
    }
}
