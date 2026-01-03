import Foundation
import MetalPerformanceShadersGraph

public enum AttentionImplementation: Sendable {
    /// Reference attention: matmul(Q, K^T) -> scale -> softmax -> matmul(attn, V)
    case referenceSoftmax

    /// Uses MPSGraph's fused scaled dot-product attention (FlashAttention-style).
    case scaledDotProduct

    /// Prefer scaledDotProduct when available; otherwise fall back to referenceSoftmax.
    case auto
}

public enum AttentionKernelError: Error {
    case unavailableScaledDotProductAttention
}

public struct AttentionKernels {
    /// Builds attention output for tensors shaped [B, H, N, D].
    public static func scaledDotProductAttentionOrReference(
        graph: MPSGraph,
        query q: MPSGraphTensor,
        key k: MPSGraphTensor,
        value v: MPSGraphTensor,
        scale: Float,
        name: String,
        implementation: AttentionImplementation
    ) -> MPSGraphTensor {
        switch implementation {
        case .referenceSoftmax:
            return referenceSoftmax(graph: graph, query: q, key: k, value: v, scale: scale, name: name)
        case .scaledDotProduct:
            if #available(macOS 15.0, iOS 18.0, *) {
                return graph.scaledDotProductAttention(query: q, key: k, value: v, mask: nil, scale: scale, name: name)
            } else {
                return referenceSoftmax(graph: graph, query: q, key: k, value: v, scale: scale, name: name)
            }
        case .auto:
            if #available(macOS 15.0, iOS 18.0, *) {
                return graph.scaledDotProductAttention(query: q, key: k, value: v, mask: nil, scale: scale, name: name)
            } else {
                return referenceSoftmax(graph: graph, query: q, key: k, value: v, scale: scale, name: name)
            }
        }
    }

    private static func referenceSoftmax(
        graph: MPSGraph,
        query q: MPSGraphTensor,
        key k: MPSGraphTensor,
        value v: MPSGraphTensor,
        scale: Float,
        name: String
    ) -> MPSGraphTensor {
        // scores: [B, H, N, N]
        let kT = graph.transposeTensor(k, dimension: 2, withDimension: 3, name: "\(name).kT")
        var scores = graph.matrixMultiplication(primary: q, secondary: kT, name: "\(name).scores")

        let scaleConst = graph.constant(Double(scale), dataType: .float32)
        scores = graph.multiplication(scores, scaleConst, name: "\(name).scale")

        // softmax over last dim
        let probs = graph.softMax(with: scores, axis: -1, name: "\(name).softmax")

        // out: [B, H, N, D]
        return graph.matrixMultiplication(primary: probs, secondary: v, name: "\(name).out")
    }
}
