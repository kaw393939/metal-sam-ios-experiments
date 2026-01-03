import Foundation

/// Single source of truth for encoder topology and IO sizes.
///
/// Defaults match the SAM3 checkpoint as defined by `WeightMapper`.
public struct SAM3EncoderConfig: Sendable, Hashable {
    public let embedDim: Int
    public let numHeads: Int
    public let numBlocks: Int
    public let patchSize: Int
    public let inputSize: Int
    public let mlpHiddenDim: Int
    public let inChannels: Int
    public let posEmbedGridSize: Int

    public init(
        embedDim: Int,
        numHeads: Int,
        numBlocks: Int,
        patchSize: Int,
        inputSize: Int,
        mlpHiddenDim: Int,
        inChannels: Int = 3,
        posEmbedGridSize: Int = 24
    ) {
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.numBlocks = numBlocks
        self.patchSize = patchSize
        self.inputSize = inputSize
        self.mlpHiddenDim = mlpHiddenDim
        self.inChannels = inChannels
        self.posEmbedGridSize = posEmbedGridSize
    }

    public var dimPerHead: Int { embedDim / numHeads }

    /// Patch grid size for the checkpoint input (e.g. 1008/14 = 64).
    public var checkpointGridSize: Int { inputSize / patchSize }

    /// Spatial token count for the checkpoint grid (e.g. 64*64 = 4096).
    public var spatialTokenCount: Int { checkpointGridSize * checkpointGridSize }

    /// Window size used by checkpoint positional embedding and windowed attention.
    public var windowSize: Int { posEmbedGridSize }

    /// Spatial token count for windowed attention / pos-embed grid (e.g. 24*24 = 576).
    public var windowTokenCount: Int { posEmbedGridSize * posEmbedGridSize }

    /// Sequence length of the stored checkpoint pos_embed tensor (e.g. 1 + 24*24 = 577).
    public var posEmbedSeqLen: Int { 1 + windowTokenCount }

    public static var sam3Checkpoint: SAM3EncoderConfig {
        SAM3EncoderConfig(
            embedDim: WeightMapper.embedDim,
            numHeads: WeightMapper.numHeads,
            numBlocks: WeightMapper.numBlocks,
            patchSize: WeightMapper.patchSize,
            inputSize: WeightMapper.inputSize,
            mlpHiddenDim: WeightMapper.mlpHiddenDim,
            inChannels: 3
        )
    }
}
