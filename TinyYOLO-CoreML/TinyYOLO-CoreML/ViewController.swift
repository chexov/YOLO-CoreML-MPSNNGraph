import UIKit
import Vision
import AVFoundation
import CoreMedia
import VideoToolbox
import SwiftOCR

class ViewController: UIViewController {
    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var timeLabel: UILabel!
    @IBOutlet weak var debugImageView: UIImageView!

    let yolo = YOLO()
    let swiftOCRInstance = SwiftOCR()

    var videoCapture: VideoCapture!
    var request: VNCoreMLRequest!
    var startTimes: [CFTimeInterval] = []

    var boundingBoxes = [BoundingBox]()
    var colors: [UIColor] = []

    let ciContext = CIContext()
    var resizedPixelBuffer: CVPixelBuffer?
    var uaplatesPixelBuffer: CVPixelBuffer?

    var framesDone = 0
    var frameCapturingStartTime = CACurrentMediaTime()
    let semaphore = DispatchSemaphore(value: 2)
    
    let uaplatesWidth = 200
    let uaplatesHeight = 84
    let uaplatesmodel = uaplates()

    override func viewDidLoad() {
        super.viewDidLoad()

        timeLabel.text = ""

        setUpBoundingBoxes()
        setUpCoreImage()
        setUpVision()
        setUpCamera()

        frameCapturingStartTime = CACurrentMediaTime()
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        print(#function)
    }

    // MARK: - Initialization

    func setUpBoundingBoxes() {
        for _ in 0..<YOLO.maxBoundingBoxes {
            boundingBoxes.append(BoundingBox())
        }

        // Make colors for the bounding boxes. There is one color for each class,
        // 20 classes in total.
        for r:CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
            for g:CGFloat in [0.3, 0.7] {
                for b:CGFloat in [0.4, 0.8] {
                    let color = UIColor(red: r, green: g, blue: b, alpha: 1)
                    colors.append(color)
                }
            }
        }
        colors[0] = UIColor.orange;
    }

    func setUpCoreImage() {
        let status = CVPixelBufferCreate(nil, YOLO.inputWidth, YOLO.inputHeight,
                kCVPixelFormatType_32BGRA, nil,
                &resizedPixelBuffer)
        if status != kCVReturnSuccess {
            print("Error: could not create resized pixel buffer", status)
        }
    }

    func setUpVision() {
        guard let visionModel = try? VNCoreMLModel(for: yolo.model.model) else {
            print("Error: could not create Vision model")
            return
        }

        request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)

        // NOTE: If you choose another crop/scale option, then you must also
        // change how the BoundingBox objects get scaled when they are drawn.
        // Currently they assume the full input image is used.
        request.imageCropAndScaleOption = .scaleFill
    }

    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 50
        videoCapture.setUp(sessionPreset: AVCaptureSession.Preset.high) { success in
            if success {
                // Add the video preview into the UI.
                if let previewLayer = self.videoCapture.previewLayer {
                    self.videoPreview.layer.addSublayer(previewLayer)
                    self.resizePreviewLayer()
                }

                // Add the bounding box layers to the UI, on top of the video preview.
                for box in self.boundingBoxes {
                    box.addToLayer(self.videoPreview.layer)
                }

                // Once everything is set up, we can start capturing live video.
                self.videoCapture.start()
            }
        }
    }

    // MARK: - UI stuff

    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        resizePreviewLayer()
    }

    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }

    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }

    // MARK: - Doing inference

    func predict(image: UIImage) {
        if let pixelBuffer = image.pixelBuffer(width: YOLO.inputWidth, height: YOLO.inputHeight) {
            predict(pixelBuffer: pixelBuffer)
        }
    }

    func predictOCR(pixelBuffer: CVPixelBuffer) {
        // Measure how long it takes to predict a single video frame.
        let startTime = CACurrentMediaTime()


        let ciImage = CIImage.init(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
//        let ocrImg = OCRImage.init(ciImage: ciImage)
//        let imgForOcr = self.swiftOCRInstance.preprocessImageForOCR(ocrImg)

        let cgImage: CGImage = self.ciContext.createCGImage(ciImage, from: ciImage.extent)!
        let uiImage = UIImage.init(cgImage: cgImage)
//        let gray = convertToGrayscale(image: uiImage)
//        let handler = VNImageRequestHandler(cgImage: gray.cgImage!)

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)

        let request: VNDetectTextRectanglesRequest =
                VNDetectTextRectanglesRequest(completionHandler: { (request, error) in
                    if (error != nil) {
                        print("Got Error In Run Text Detect Request :(")
                    } else {
                        guard let results = request.results as? Array<VNTextObservation> else {
                            fatalError("Unexpected result type from VNDetectTextRectanglesRequest")
                        }
                        if (results.count == 0) {
                            self.semaphore.signal()
                            return
                        }

//                        print("results " + String(results.count))
                        var predictions = [YOLO.Prediction]()
                        for textObservation in results {
//                            let cropped: CIImage = ciImage.cropped(to: textObservation.boundingBox)
//                            let ocrImage = OCRImage.init(ciImage: cropped)
////                            let gray = convertToGrayscale(image: cropped)
//                            let ocrImage2 = self.swiftOCRInstance.preprocessImageForOCR(ocrImage)
////
//                            self.swiftOCRInstance.recognizeInRect(ocrImage2,
//                                    rect: textObservation.boundingBox,
//                                    completionHandler: { recognizedString in
//                                        print(recognizedString)
//                                    })


                            let textrect = self.expandBoundingBox(textObservation: textObservation, imgWidth: CVPixelBufferGetWidth(pixelBuffer), imgHeight: CVPixelBufferGetHeight(pixelBuffer))
                            
                            print(String(textrect.minX.native) + " " + String(textrect.minY.native) + " " + String(textrect.width.native) + " " + String(textrect.height.native))
                            
                            let cropped: CIImage = ciImage.cropped(to: textrect)
                            let croppedPixelbuffer = uiImage.pixelBuffer(width: Int(textrect.width), height: Int(textrect.height))
                            context.render(cropped, to: croppedPixelbuffer!)

                            // Resize the input with Core Image to 200x84.
                            let resizedPixelBuffer = resizePixelBuffer(croppedPixelbuffer!,
                                                                          width: self.uaplatesWidth,
                                                                          height: self.uaplatesHeight)
                            
                            if let modelOutput = try? self.uaplatesmodel.prediction(image: resizedPixelBuffer!) {
                                var label = "X3"
                                if (modelOutput.output1[0].doubleValue > 0.5) {
                                    label = "plate"
                                }
                                print("isPlate = " + label)
                                let prediction = YOLO.Prediction(classIndex: 0,
                                                                 score: Float(modelOutput.output1[0].doubleValue),
                                                                 rect: textObservation.boundingBox,
                                                                 ocr: label)
                                predictions.append(prediction)
                            }
                        }

                        let elapsed = CACurrentMediaTime() - startTime
                        self.showOnMainThread(predictions, elapsed)
                    }
                })
        request.reportCharacterBoxes = true
        do {
            try handler.perform([request])
        } catch {
            print(error)
        }
    }
    
    func expandBoundingBox(textObservation: VNTextObservation, imgWidth: Int, imgHeight: Int) -> CGRect {
        let x = textObservation.boundingBox.origin.x * CGFloat(imgWidth)
        let y = textObservation.boundingBox.origin.y * CGFloat(imgHeight)
        let width = textObservation.boundingBox.size.width * CGFloat(imgWidth)
        let height = textObservation.boundingBox.size.height * CGFloat(imgHeight)
        
        var rect = CGRect()
        rect.origin.x = x
        rect.origin.y = y
        rect.size.width = width
        rect.size.height = height
        
        
        if (rect.minX - 25 < 0 || rect.minY - 25 < 0) {
            return rect
        }
        
        let expandedRect = CGRect(x: rect.minX - 25,
                          y: rect.minY - 25,
                          width: rect.width + 50,
                          height: rect.height + 50)
        return expandedRect
    }

    func predict(pixelBuffer: CVPixelBuffer) {
        // Measure how long it takes to predict a single video frame.
        let startTime = CACurrentMediaTime()

        // Resize the input with Core Image to 416x416.
        guard let resizedPixelBuffer = resizedPixelBuffer else {
            return
        }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let sx = CGFloat(YOLO.inputWidth) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let sy = CGFloat(YOLO.inputHeight) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
        let scaledImage = ciImage.transformed(by: scaleTransform)
        ciContext.render(scaledImage, to: resizedPixelBuffer)

        // This is an alternative way to resize the image (using vImage):
        //if let resizedPixelBuffer = resizePixelBuffer(pixelBuffer,
        //                                              width: YOLO.inputWidth,
        //                                              height: YOLO.inputHeight)

        // Resize the input to 416x416 and give it to our model.
        if let boundingBoxes = try? yolo.predict(image: resizedPixelBuffer) {
            let elapsed = CACurrentMediaTime() - startTime
            showOnMainThread(boundingBoxes, elapsed)
        }
    }

    func predictUsingVision(pixelBuffer: CVPixelBuffer) {
        // Measure how long it takes to predict a single video frame. Note that
        // predict() can be called on the next frame while the previous one is
        // still being processed. Hence the need to queue up the start times.
        startTimes.append(CACurrentMediaTime())

        // Vision will automatically resize the input image.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
    }

    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
           let features = observations.first?.featureValue.multiArrayValue {

            let boundingBoxes = yolo.computeBoundingBoxes(features: features)
            let elapsed = CACurrentMediaTime() - startTimes.remove(at: 0)
            showOnMainThread(boundingBoxes, elapsed)
        }
    }

    func showOnMainThread(_ boundingBoxes: [YOLO.Prediction], _ elapsed: CFTimeInterval) {
        DispatchQueue.main.async {
//             For debugging, to make sure the resized CVPixelBuffer is correct.
//            var debugImage: CGImage?
//            VTCreateCGImageFromCVPixelBuffer(self.resizedPixelBuffer!, nil, &debugImage)
//            self.debugImageView.image = UIImage(cgImage: debugImage!)

            self.show(predictions: boundingBoxes)

            let fps = self.measureFPS()
            self.timeLabel.text = String(format: "Elapsed %.5f seconds - %.2f FPS", elapsed, fps)

            self.semaphore.signal()
        }
    }

    func measureFPS() -> Double {
        // Measure how many frames were actually delivered per second.
        framesDone += 1
        let frameCapturingElapsed = CACurrentMediaTime() - frameCapturingStartTime
        let currentFPSDelivered = Double(framesDone) / frameCapturingElapsed
        if frameCapturingElapsed > 1 {
            framesDone = 0
            frameCapturingStartTime = CACurrentMediaTime()
        }
        return currentFPSDelivered
    }

    func show(predictions: [YOLO.Prediction]) {
        for i in 0..<boundingBoxes.count {
            if i < predictions.count {
                let prediction = predictions[i]

                var pred = prediction.rect
                pred.size.width = view.bounds.width * pred.size.width
                pred.size.height = view.bounds.height * pred.size.height
                pred.origin.x = view.bounds.width * pred.origin.x
                pred.origin.y = view.bounds.height * pred.origin.y
                pred.origin.y = view.bounds.height - pred.origin.y - pred.height

//                print(String(format: "x: %1.f y: %1.f, w:%1.f, h:%1.f", pred.origin.x, pred.origin.y, pred.size.width, pred.size.height))

                let label = String(format: "%@ %1.f", prediction.ocr, prediction.score * 100)
                let color = colors[prediction.classIndex]
                boundingBoxes[i].show(frame: pred, label: label, color: color)
            } else {
                boundingBoxes[i].hide()
            }
        }
    }
}


func getRectAbs(textObservation: VNTextObservation, imgWidth: Int, imgHeight: Int) -> CGRect {
    let x = textObservation.boundingBox.origin.x * CGFloat(imgWidth)
    let y = textObservation.boundingBox.origin.y * CGFloat(imgHeight)
    let width = textObservation.boundingBox.size.width * CGFloat(imgWidth)
    let height = textObservation.boundingBox.size.height * CGFloat(imgHeight)

    var rect = CGRect()
    rect.origin.x = x
    rect.origin.y = y
    rect.size.width = width
    rect.size.height = height

//    let s = rect.minX.description + " " + rect.minY.description + " " + rect.maxX.description + " " + rect.maxY.description
//    print(s)
    return rect
}

func getRectRelative(textObservation: VNTextObservation) -> CGRect {
    let x = textObservation.boundingBox.origin.x
    let y = textObservation.boundingBox.origin.y
    let width = textObservation.boundingBox.size.width
    let height = textObservation.boundingBox.size.height

    var rect = CGRect()
    rect.origin.x = x
    rect.origin.y = y
    rect.size.width = width
    rect.size.height = height

//    let s = rect.minX.description + " " + rect.minY.description + " " + rect.maxX.description + " " + rect.maxY.description
//    print(s)
    return rect
}

extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {

        // For debugging.
//        let image = UIImage(named: "dog416")!
//        if let pixelBuffer = image.pixelBuffer(width: 1140, height: 2106) {
//            predictOCR(pixelBuffer: pixelBuffer);
//            let ciImage = CIImage.init(cvImageBuffer: pixelBuffer)
//            self.ciContext.render(ciImage, to: self.resizedPixelBuffer!)
//            return
//        }

        semaphore.wait()

        if let pixelBuffer = pixelBuffer {
            // For better throughput, perform the prediction on a background queue
            // instead of on the VideoCapture queue. We use the semaphore to block
            // the capture queue and drop frames when Core ML can't keep up.
            DispatchQueue.global().async {
//                self.predict(pixelBuffer: pixelBuffer)
//                self.predictUsingVision(pixelBuffer: pixelBuffer)

                self.predictOCR(pixelBuffer: pixelBuffer)
//
//                let ciImage = CIImage.init(cvImageBuffer: pixelBuffer)
//                self.ciContext.render(ciImage, to: self.resizedPixelBuffer!)


            }
        }
    }
}
