import Foundation
import UIKit
import CoreML
import Accelerate

// The labels for the 20 classes.
let labels = [
  "UA", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
  "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
  "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

let anchors: [Float] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

/**
  Removes bounding boxes that overlap too much with other boxes that have
  a higher score.

  Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc

  - Parameters:
    - boxes: an array of bounding boxes and their scores
    - limit: the maximum number of boxes that will be selected
    - threshold: used to decide whether boxes overlap too much
*/
func nonMaxSuppression(boxes: [YOLO.Prediction], limit: Int, threshold: Float) -> [YOLO.Prediction] {

  // Do an argsort on the confidence scores, from high to low.
  let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }

  var selected: [YOLO.Prediction] = []
  var active = [Bool](repeating: true, count: boxes.count)
  var numActive = active.count

  // The algorithm is simple: Start with the box that has the highest score.
  // Remove any remaining boxes that overlap it more than the given threshold
  // amount. If there are any boxes left (i.e. these did not overlap with any
  // previous boxes), then repeat this procedure, until no more boxes remain
  // or the limit has been reached.
  outer: for i in 0..<boxes.count {
    if active[i] {
      let boxA = boxes[sortedIndices[i]]
      selected.append(boxA)
      if selected.count >= limit { break }

      for j in i+1..<boxes.count {
        if active[j] {
          let boxB = boxes[sortedIndices[j]]
          if IOU(a: boxA.rect, b: boxB.rect) > threshold {
            active[j] = false
            numActive -= 1
            if numActive <= 0 { break outer }
          }
        }
      }
    }
  }
  return selected
}

/**
  Computes intersection-over-union overlap between two bounding boxes.
*/
public func IOU(a: CGRect, b: CGRect) -> Float {
  let areaA = a.width * a.height
  if areaA <= 0 { return 0 }

  let areaB = b.width * b.height
  if areaB <= 0 { return 0 }

  let intersectionMinX = max(a.minX, b.minX)
  let intersectionMinY = max(a.minY, b.minY)
  let intersectionMaxX = min(a.maxX, b.maxX)
  let intersectionMaxY = min(a.maxY, b.maxY)
  let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                         max(intersectionMaxX - intersectionMinX, 0)
  return Float(intersectionArea / (areaA + areaB - intersectionArea))
}

extension Array where Element: Comparable {
  /**
    Returns the index and value of the largest element in the array.
  */
  public func argmax() -> (Int, Element) {
    precondition(self.count > 0)
    var maxIndex = 0
    var maxValue = self[0]
    for i in 1..<self.count {
      if self[i] > maxValue {
        maxValue = self[i]
        maxIndex = i
      }
    }
    return (maxIndex, maxValue)
  }
}

/**
  Logistic sigmoid.
*/
public func sigmoid(_ x: Float) -> Float {
  return 1 / (1 + exp(-x))
}

/**
  Computes the "softmax" function over an array.

  Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/

  This is what softmax looks like in "pseudocode" (actually using Python
  and numpy):

      x -= np.max(x)
      exp_scores = np.exp(x)
      softmax = exp_scores / np.sum(exp_scores)

  First we shift the values of x so that the highest value in the array is 0.
  This ensures numerical stability with the exponents, so they don't blow up.
*/
public func softmax(_ x: [Float]) -> [Float] {
  var x = x
  let len = vDSP_Length(x.count)

  // Find the maximum value in the input array.
  var max: Float = 0
  vDSP_maxv(x, 1, &max, len)

  // Subtract the maximum from all the elements in the array.
  // Now the highest value in the array is 0.
  max = -max
  vDSP_vsadd(x, 1, &max, &x, 1, len)

  // Exponentiate all the elements in the array.
  var count = Int32(x.count)
  vvexpf(&x, x, &count)

  // Compute the sum of all exponentiated values.
  var sum: Float = 0
  vDSP_sve(x, 1, &sum, len)

  // Divide each element by the sum. This normalizes the array contents
  // so that they all add up to 1.
  vDSP_vsdiv(x, 1, &sum, &x, 1, len)

  return x
}



import Foundation
import UIKit
import Vision

precedencegroup ForwardPipe {
  associativity: left
  higherThan: LogicalConjunctionPrecedence
}

infix operator |> : ForwardPipe

/// Swift implementation of the forward pipe operator from F#.
///
/// Used for better readibility when piping results of one function to the next ones.
/// More details here: https://goo.gl/nHzeS6.
public func |> <T, U>(value: T, function: ((T) -> U)) -> U {
  return function(value)
}

func resize(image: UIImage, targetSize: CGSize) -> UIImage {
  let rect = CGRect(x: 0, y: 0, width: targetSize.width, height: targetSize.height)
  UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
  image.draw(in: rect)
  let newImage = UIGraphicsGetImageFromCurrentImageContext()
  UIGraphicsEndImageContext()
  return newImage!
}

func convertToGrayscale(image: UIImage) -> UIImage {
  let colorSpace: CGColorSpace = CGColorSpaceCreateDeviceGray()
  let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
  let context = CGContext(data: nil,
          width: Int(UInt(image.size.width)),
          height: Int(UInt(image.size.height)),
          bitsPerComponent: 8,
          bytesPerRow: 0,
          space: colorSpace,
          bitmapInfo: bitmapInfo.rawValue)
  context?.draw(image.cgImage!,
          in: CGRect(x: 0.0, y: 0.0, width: image.size.width, height: image.size.height))
  let imageRef: CGImage = context!.makeImage()!
  let newImage: UIImage = UIImage(cgImage: imageRef)
  return newImage
}

func insertInsets(image: UIImage, insetWidthDimension: CGFloat, insetHeightDimension: CGFloat)
                -> UIImage {
  let adjustedImage = adjustColors(image: image)
  let upperLeftPoint: CGPoint = CGPoint(x: 0, y: 0)
  let lowerLeftPoint: CGPoint = CGPoint(x: 0, y: adjustedImage.size.height - 1)
  let upperRightPoint: CGPoint = CGPoint(x: adjustedImage.size.width - 1, y: 0)
  let lowerRightPoint: CGPoint = CGPoint(x: adjustedImage.size.width - 1,
          y: adjustedImage.size.height - 1)
  let upperLeftColor: UIColor = getPixelColor(fromImage: adjustedImage, pixel: upperLeftPoint)
  let lowerLeftColor: UIColor = getPixelColor(fromImage: adjustedImage, pixel: lowerLeftPoint)
  let upperRightColor: UIColor = getPixelColor(fromImage: adjustedImage, pixel: upperRightPoint)
  let lowerRightColor: UIColor = getPixelColor(fromImage: adjustedImage, pixel: lowerRightPoint)
  let color =
          averageColor(fromColors: [upperLeftColor, lowerLeftColor, upperRightColor, lowerRightColor])
  let insets = UIEdgeInsets(top: insetHeightDimension,
          left: insetWidthDimension,
          bottom: insetHeightDimension,
          right: insetWidthDimension)
  let size = CGSize(width: adjustedImage.size.width + insets.left + insets.right,
          height: adjustedImage.size.height + insets.top + insets.bottom)
  UIGraphicsBeginImageContextWithOptions(size, false, adjustedImage.scale)
  let origin = CGPoint(x: insets.left, y: insets.top)
  adjustedImage.draw(at: origin)
  let imageWithInsets = UIGraphicsGetImageFromCurrentImageContext()
  UIGraphicsEndImageContext()
  return convertTransparent(image: imageWithInsets!, color: color)
}

func averageColor(fromColors colors: [UIColor]) -> UIColor {
  var averages = [CGFloat]()
  for i in 0..<4 {
    var total: CGFloat = 0
    for j in 0..<colors.count {
      let current = colors[j]
      let value = CGFloat(current.cgColor.components![i])
      total += value
    }
    let avg = total / CGFloat(colors.count)
    averages.append(avg)
  }
  return UIColor(red: averages[0], green: averages[1], blue: averages[2], alpha: averages[3])
}

func adjustColors(image: UIImage) -> UIImage {
  let context = CIContext(options: nil)
  if let currentFilter = CIFilter(name: "CIColorControls") {
    let beginImage = CIImage(image: image)
    currentFilter.setValue(beginImage, forKey: kCIInputImageKey)
    currentFilter.setValue(0, forKey: kCIInputSaturationKey)
    currentFilter.setValue(1.45, forKey: kCIInputContrastKey) //previous 1.5
    if let output = currentFilter.outputImage {
      if let cgimg = context.createCGImage(output, from: output.extent) {
        let processedImage = UIImage(cgImage: cgimg)
        return processedImage
      }
    }
  }
  return image
}

func fixOrientation(image: UIImage) -> UIImage {
  if image.imageOrientation == UIImageOrientation.up {
    return image
  }
  UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
  image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
  if let normalizedImage: UIImage = UIGraphicsGetImageFromCurrentImageContext() {
    UIGraphicsEndImageContext()
    return normalizedImage
  } else {
    return image
  }
}

func convertTransparent(image: UIImage, color: UIColor) -> UIImage {
  UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
  let width = image.size.width
  let height = image.size.height
  let imageRect: CGRect = CGRect(x: 0.0, y: 0.0, width: width, height: height)
  let ctx: CGContext = UIGraphicsGetCurrentContext()!
  let redValue = CGFloat(color.cgColor.components![0])
  let greenValue = CGFloat(color.cgColor.components![1])
  let blueValue = CGFloat(color.cgColor.components![2])
  let alphaValue = CGFloat(color.cgColor.components![3])
  ctx.setFillColor(red: redValue, green: greenValue, blue: blueValue, alpha: alphaValue)
  ctx.fill(imageRect)
  image.draw(in: imageRect)
  let newImage: UIImage = UIGraphicsGetImageFromCurrentImageContext()!
  UIGraphicsEndImageContext()
  return newImage
}

func getPixelColor(fromImage image: UIImage, pixel: CGPoint) -> UIColor {
  let pixelData = image.cgImage!.dataProvider!.data
  let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
  let pixelInfo: Int = ((Int(image.size.width) * Int(pixel.y)) + Int(pixel.x)) * 4
  let r = CGFloat(data[pixelInfo]) / CGFloat(255.0)
  let g = CGFloat(data[pixelInfo + 1]) / CGFloat(255.0)
  let b = CGFloat(data[pixelInfo + 2]) / CGFloat(255.0)
  let a = CGFloat(data[pixelInfo + 3]) / CGFloat(255.0)
  return UIColor(red: r, green: g, blue: b, alpha: a)
}

func crop(image: UIImage, boundingBox: CGRect) -> UIImage? {
  var t: CGAffineTransform = CGAffineTransform.identity;
  t = t.scaledBy(x: image.size.width, y: -image.size.height);
  t = t.translatedBy(x: 0, y: -1 );
  let x = boundingBox.applying(t).origin.x
  let y = boundingBox.applying(t).origin.y
  let width = boundingBox.applying(t).width
  let height = boundingBox.applying(t).height
  let fromRect = CGRect(x: x-32, y: y, width: width+32, height: height)
  let drawImage = image.cgImage!.cropping(to: fromRect)
  if let drawImage = drawImage {
    let uiImage = UIImage(cgImage: drawImage)
    return uiImage
  }
  return nil
}

func preProcess(image: UIImage) -> UIImage {
  let width = image.size.width
  let height = image.size.height
  let addToHeight2 = height / 2
  let addToWidth2 = ((6 * height) / 3 - width) / 2
  let imageWithInsets = insertInsets(image: image,
          insetWidthDimension: addToWidth2,
          insetHeightDimension: addToHeight2)
  let size = CGSize(width: 28, height: 28)
  let resizedImage = resize(image: imageWithInsets, targetSize: size)
  let grayScaleImage = convertToGrayscale(image: resizedImage)
  return grayScaleImage
}
