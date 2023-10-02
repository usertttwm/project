public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private static final int TF_OD_API_INPUT_SIZE = 1024;
    private static final boolean TF_OD_API_IS_QUANTIZED = false;
    private static final String TF_OD_API_MODEL_FILE = "trayyolo.tflite";


    private static final String TF_OD_API_LABELS_FILE = "customclasses.txt";
    private static final int TF_OD_API_INPUT_SIZEE = 608;

    private static final String TF_OD_API_MODEL_FILEE = "vggclass.tflite";



    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.3f;
    private static final boolean MAINTAIN_ASPECT = true;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 640);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private YoloV5Classifier detector;
    private yeni detector2;


    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private Bitmap xyBitmap = null;
    private Bitmap copyOfOriginalBitmap = null;


    private boolean computingDetection = false;


    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;
    private MultiBoxTracker2 tracker2;

    private BorderedText borderedText;



    private TFLiteObjectDetectionAPIModel detector1;


    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);
        tracker2 = new MultiBoxTracker2(this);

        final int modelIndex = modelView.getCheckedItemPosition();
        final String modelString = modelStrings.get(modelIndex);

        try {
            detector = DetectorFactory.getDetector(getAssets(), modelString);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        int cropSize = detector.getInputSize();

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();
        int targetW = (int) (previewWidth / 2.0);
        int targetH = (int) (previewHeight / 2.0);
        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        xyBitmap = Bitmap.createBitmap(targetW, targetH, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
        OverlayView trackingOverlay2 = (OverlayView) findViewById(R.id.tracking_overlay);

        trackingOverlay2.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas1) {
                        tracker2.draw(canvas1);
                        if (isDebug()) {
                            tracker2.drawDebug(canvas1);
                        }
                    }
                });

        tracker2.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
        try {
            detector1 =
                    (TFLiteObjectDetectionAPIModel) TFLiteObjectDetectionAPIModel.create(
                            this,
                            TF_OD_API_MODEL_FILEE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZEE,
                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }
        try {
            final String modelString1 = modelStrings.get(6);

            detector2 = DetectorFactoryaksam.getDetector(getAssets(), modelString1);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

    }
    protected void updateActiveModel() {
        // Get UI information before delegating to background
        final int modelIndex = modelView.getCheckedItemPosition();
        final int deviceIndex = deviceView.getCheckedItemPosition();
        String threads = threadsTextView.getText().toString().trim();
        final int numThreads = Integer.parseInt(threads);

        handler.post(() -> {
            if (modelIndex == currentModel && deviceIndex == currentDevice
                    && numThreads == currentNumThreads) {
                return;
            }
            currentModel = modelIndex;
            currentDevice = deviceIndex;
            currentNumThreads = numThreads;

            // Disable classifier while updating
            if (detector != null) {
                detector.close();
                detector = null;
            }

            // Lookup names of parameters.
            String modelString = modelStrings.get(modelIndex);
            String device = deviceStrings.get(deviceIndex);
            String modelString2 = modelStrings.get(6);

            LOGGER.i("Changing model to " + modelString + " device " + device);

            // Try to load model.

            try {
                detector = DetectorFactory.getDetector(getAssets(), modelString);
                detector2 = DetectorFactoryaksam.getDetector(getAssets(), modelString2);

                // Customize the interpreter to the type of device we want to use.
                if (detector == null || detector2 == null) {
                    return;
                }
            }
            catch(IOException e) {
                e.printStackTrace();
                LOGGER.e(e, "Exception in updateActiveModel()");
                Toast toast =
                        Toast.makeText(
                                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                toast.show();
                finish();
            }


            if (device.equals("CPU")) {
                detector.useCPU();
                detector2.useCPU();

            } else if (device.equals("GPU")) {
                detector.useGpu();
                detector2.useGpu();
            } else if (device.equals("NNAPI")) {
                detector.useNNAPI();
                detector2.useNNAPI();

            }
            detector.setNumThreads(numThreads);
            detector2.setNumThreads(numThreads);

            int cropSize = detector.getInputSize();
            croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

            frameToCropTransform =
                    ImageUtils.getTransformationMatrix(
                            previewWidth, previewHeight,
                            cropSize, cropSize,
                            sensorOrientation, MAINTAIN_ASPECT);

            cropToFrameTransform = new Matrix();
            frameToCropTransform.invert(cropToFrameTransform);
        });
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);

        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(new Runnable() {
            @Override
            public void run() {
                LOGGER.i("Running detection on image " + currTimestamp);
                final long startTime = SystemClock.uptimeMillis();
                final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                Log.e("CHECK", "run: " + results.size());

                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                final Canvas canvas = new Canvas(cropCopyBitmap);

                final Paint paint = new Paint();
                paint.setColor(Color.RED);
                paint.setStyle(Style.STROKE);
                paint.setStrokeWidth(2.0f);

                float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                final List<ClassifierThree.Recognition> mappedRecognitionsmy =
                        new LinkedList<>();
                final List<Classifier.Recognition> mappedRecognitions =
                        new LinkedList<Classifier.Recognition>();
                for (final Classifier.Recognition result : results) {
                    final RectF location = result.getLocation();
                    if (location != null && result.getConfidence() >= minimumConfidence) {
                        canvas.drawRect(location, paint);

                        cropToFrameTransform.mapRect(location);
                        result.setLocation(location);
                        mappedRecognitions.add(result);

                        //CROP
                        Bitmap copyOfOriginalBitmap = rgbFrameBitmap.copy(rgbFrameBitmap.getConfig(), true);
                        Bitmap secondBitmap = Bitmap.createBitmap(copyOfOriginalBitmap,
                                (int) location.left, (int) location.top, (int) location.right - (int) location.left, (int) location.bottom - (int) location.top);
                        Bitmap secondScaledBitmap = Bitmap.createScaledBitmap(secondBitmap, 608, 608, true);


                        final String resultLabel = detector1.recognizeImage1(secondScaledBitmap);

                        Float confidence = -1f;
                        final Classifier.Recognition result1 = new Classifier.Recognition(
                                "0", resultLabel, confidence, location);

                        result1.setLocation(location);

                        mappedRecognitions.add(result1);
                        final Canvas canvas1 = new Canvas(secondScaledBitmap);

                        // Process with detector2
                        if (secondScaledBitmap != null) {
                            if (resultLabel.equals("eveningTray")) {
                                Matrix matrix = new Matrix();
                                matrix.postRotate(90);

                                Bitmap rotatedSecondScaledBitmap = Bitmap.createBitmap(secondScaledBitmap, 0, 0, secondScaledBitmap.getWidth(), secondScaledBitmap.getHeight(), matrix, true);
                                final ArrayList<ClassifierThree.Recognition> resultss = detector2.recognizeImage2(rotatedSecondScaledBitmap);

                                for (final ClassifierThree.Recognition result2 : resultss) {
                                    final RectF location2 = result2.getLocation();

                                    if (location2 != null && result2.getConfidence() >= minimumConfidence) {
                                        canvas1.drawRect(location2, paint);
                                        canvas1.rotate(90, canvas1.getWidth() / 2, canvas1.getHeight() / 2);

                                        cropToFrameTransform.mapRect(location2);
                                        result2.setLocation(location2);
                                        mappedRecognitionsmy.add(result2);
                                        tracker2.trackResults(mappedRecognitionsmy, currTimestamp);

                                    }
                                }
                            }


                        }
                    }

                }

                tracker.trackResults(mappedRecognitions, currTimestamp);

                trackingOverlay.postInvalidate();
                computingDetection = false;

                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        showFrameInfo(previewWidth + "x" + previewHeight);
                        showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                        showInference(lastProcessingTimeMs + "ms");
                    }
                });
            }
        });



    }


    @Override
    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    @Override
    protected void setUseNNAPI(final boolean isChecked) {
        runInBackground(() -> detector.setUseNNAPI(isChecked));
    }

    @Override
    protected void setNumThreads(final int numThreads) {
        runInBackground(() -> detector.setNumThreads(numThreads));
    }
}
