import Foundation
import Metal

public typealias ModelWeights = [String: Data]

public class ModelLoader {
    public enum ModelLoaderError: Error, Equatable {
        case unsupportedFileFormat(String)
        case offlinePackedWeightsRequired
        case invalidPackedWeights
    }

    public static func loadBuffer(from data: Data, device: MTLDevice, label: String? = nil) -> MTLBuffer? {
        let length = data.count
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else { return nil }
        
        data.withUnsafeBytes { ptr in
            if let baseAddress = ptr.baseAddress {
                buffer.contents().copyMemory(from: baseAddress, byteCount: length)
            }
        }
        buffer.label = label
        return buffer
    }
    
    public func load(url: URL) throws -> ModelWeights {
        var weights: ModelWeights = [:]
        
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: url.path) {
            // check if directory
            var isDir: ObjCBool = false
            fileManager.fileExists(atPath: url.path, isDirectory: &isDir)
            
            if isDir.boolValue {
                let contents = try fileManager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
                for fileURL in contents {
                    let key = fileURL.deletingPathExtension().lastPathComponent
                    let data = try Data(contentsOf: fileURL)
                    weights[key] = data
                }
            } else if url.pathExtension == "wts" {
                // Packed weights file (fast path)
                return try loadPackedWeights(url: url)
            } else if url.pathExtension == "npz" {
                // Mac App Store + performance: runtime must not spawn /usr/bin/unzip.
                // Require an offline-generated packed artifact next to the NPZ.
                let packed = url.deletingPathExtension().appendingPathExtension("wts")
                if fileManager.fileExists(atPath: packed.path) {
                    return try loadPackedWeights(url: packed)
                }
                throw ModelLoaderError.offlinePackedWeightsRequired
            } else {
                throw ModelLoaderError.unsupportedFileFormat(url.pathExtension)
            }
        }
        return weights
    }

    /// Load packed weights file.
    /// Format (little-endian):
    /// - magic: 8 bytes: "SAM3WTS1\0"
    /// - numEntries: u32
    /// Repeated entries:
    /// - keyLen: u32
    /// - key: [u8]*keyLen
    /// - dtype: u8 (opaque to loader)
    /// - rank: u8
    /// - shape: [u32]*rank
    /// - byteLen: u32
    /// - payload: [u8]*byteLen
    private func loadPackedWeights(url: URL) throws -> ModelWeights {
        let fileData = try Data(contentsOf: url, options: [.mappedIfSafe])
        var offset = 0

        func require(_ condition: Bool) throws {
            if !condition { throw ModelLoaderError.invalidPackedWeights }
        }

        func readBytes(_ count: Int) throws -> Data {
            try require(offset + count <= fileData.count)
            let out = fileData.subdata(in: offset..<(offset + count))
            offset += count
            return out
        }

        func readU8() throws -> UInt8 {
            let d = try readBytes(1)
            return d[d.startIndex]
        }

        func readU32() throws -> UInt32 {
            let d = try readBytes(4)
            return d.withUnsafeBytes { raw in
                raw.load(as: UInt32.self).littleEndian
            }
        }

        let baseAddress: UnsafeRawPointer = try fileData.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { throw ModelLoaderError.invalidPackedWeights }
            return base
        }

        // magic
        let magic = try readBytes(8)
        let expectedMagic = Data([0x53, 0x41, 0x4D, 0x33, 0x57, 0x54, 0x53, 0x31]) // "SAM3WTS1"
        try require(magic == expectedMagic)

        let numEntries = Int(try readU32())
        try require(numEntries >= 0)

        var weights: ModelWeights = [:]
        weights.reserveCapacity(numEntries)

        for _ in 0..<numEntries {
            let keyLen = Int(try readU32())
            try require(keyLen > 0)
            let keyData = try readBytes(keyLen)
            guard let key = String(data: keyData, encoding: .utf8) else {
                throw ModelLoaderError.invalidPackedWeights
            }

            _ = try readU8() // dtype (currently not needed)
            let rank = Int(try readU8())
            try require(rank >= 0 && rank <= 8)
            if rank > 0 {
                _ = try readBytes(rank * 4) // shape dims
            }

            let byteLen = Int(try readU32())
            try require(byteLen >= 0)

            // Zero-copy payload: create a Data view into the mapped file.
            try require(offset + byteLen <= fileData.count)
            let payloadPtr = UnsafeMutableRawPointer(mutating: baseAddress.advanced(by: offset))
            let payload = Data(bytesNoCopy: payloadPtr, count: byteLen, deallocator: .custom { _, _ in
                // Keep the underlying mapped file alive until this Data is released.
                _ = fileData
            })
            offset += byteLen
            weights[key] = payload
        }

        return weights
    }
    
}

// Helper extension for components
extension Dictionary where Key == String, Value == Data {
    func buffer(for key: String, device: MTLDevice) -> MTLBuffer? {
        if let data = self[key] {
            return ModelLoader.loadBuffer(from: data, device: device, label: key)
        }
        // Try common prefixes
        if let data = self["mask_decoder." + key] {
            return ModelLoader.loadBuffer(from: data, device: device, label: "mask_decoder." + key)
        }
        return nil
    }
}
