import Foundation
import CoreGraphics
import ImageIO

public struct IoUScore: Sendable {
    public let intersection: Int
    public let union: Int

    public var iou: Double {
        guard union > 0 else { return 1.0 }
        return Double(intersection) / Double(union)
    }
}

public enum IoUError: Error {
    case cannotLoadImage
    case sizeMismatch
    case cannotCreateContext
}

public enum IoUMetrics {
    public static func loadCGImage(from url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw IoUError.cannotLoadImage
        }
        return img
    }

    /// Computes IoU between two images interpreted as binary masks.
    /// - Threshold is applied to 8-bit grayscale (0..255). Values >= threshold count as 1.
    public static func iou(pred: CGImage, gt: CGImage, threshold: UInt8 = 128) throws -> IoUScore {
        guard pred.width == gt.width, pred.height == gt.height else {
            throw IoUError.sizeMismatch
        }

        let w = pred.width
        let h = pred.height

        let predGray = try grayscale8(pred)
        let gtGray = try grayscale8(gt)

        var intersection = 0
        var union = 0

        predGray.withUnsafeBytes { (pBytes: UnsafeRawBufferPointer) in
            gtGray.withUnsafeBytes { (gBytes: UnsafeRawBufferPointer) in
                let p = pBytes.bindMemory(to: UInt8.self)
                let g = gBytes.bindMemory(to: UInt8.self)
                let t = threshold

                for i in 0..<(w * h) {
                    let pb = p[i] >= t
                    let gb = g[i] >= t
                    if pb && gb { intersection += 1 }
                    if pb || gb { union += 1 }
                }
            }
        }

        return IoUScore(intersection: intersection, union: union)
    }

    private static func grayscale8(_ img: CGImage) throws -> Data {
        let w = img.width
        let h = img.height

        var data = Data(count: w * h)
        let colorSpace = CGColorSpaceCreateDeviceGray()

        let ok = data.withUnsafeMutableBytes { (bytes: UnsafeMutableRawBufferPointer) -> Bool in
            guard let base = bytes.baseAddress else { return false }
            guard let ctx = CGContext(
                data: base,
                width: w,
                height: h,
                bitsPerComponent: 8,
                bytesPerRow: w,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) else {
                return false
            }

            ctx.interpolationQuality = .none
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }

        if !ok {
            throw IoUError.cannotCreateContext
        }

        return data
    }
}
