// Copyright © 2024 Apple Inc.

import Foundation
import MLX

// MARK: - Audio Configuration

public struct Gemma3nAudioConfiguration: Codable, Sendable {
    public let inputFeatSize: Int
    public let hiddenSize: Int
    public let confAttentionChunkSize: Int
    public let confAttentionContextLeft: Int
    public let confAttentionContextRight: Int
    public let confAttentionInvalidLogitsValue: Float?
    public let confAttentionLogitCap: Float
    public let confNumAttentionHeads: Int
    public let confNumHiddenLayers: Int
    public let confConvKernelSize: Int
    public let confPositionalBiasSize: Int?
    public let confReductionFactor: Int
    public let confResidualWeight: Float
    public let sscpConvChannelSize: [Int]
    public let sscpConvGroupNormEps: Float
    public let sscpConvKernelSize: [[Int]]
    public let sscpConvStrideSize: [[Int]]
    public let vocabSize: Int
    public let sscpConvEps: Float?
    public let rmsNormEps: Float
    public let gradientClipping: Float
    public let vocabOffset: Int
    
    public init(
        inputFeatSize: Int = 80,
        hiddenSize: Int = 1536,
        confAttentionChunkSize: Int = 12,
        confAttentionContextLeft: Int = 13,
        confAttentionContextRight: Int = 0,
        confAttentionInvalidLogitsValue: Float? = -1e9,
        confAttentionLogitCap: Float = 50.0,
        confNumAttentionHeads: Int = 8,
        confNumHiddenLayers: Int = 12,
        confConvKernelSize: Int = 5,
        confPositionalBiasSize: Int? = 256,
        confReductionFactor: Int = 4,
        confResidualWeight: Float = 0.5,
        sscpConvChannelSize: [Int] = [128, 32],
        sscpConvGroupNormEps: Float = 1e-3,
        sscpConvKernelSize: [[Int]] = [[3, 3], [3, 3]],
        sscpConvStrideSize: [[Int]] = [[2, 2], [2, 2]],
        vocabSize: Int = 128,
        sscpConvEps: Float? = 1e-3,
        rmsNormEps: Float = 1e-6,
        gradientClipping: Float = 10000000000.0,
        vocabOffset: Int = 262_144 + 128
    ) {
        self.inputFeatSize = inputFeatSize
        self.hiddenSize = hiddenSize
        self.confAttentionChunkSize = confAttentionChunkSize
        self.confAttentionContextLeft = confAttentionContextLeft
        self.confAttentionContextRight = confAttentionContextRight
        self.confAttentionInvalidLogitsValue = confAttentionInvalidLogitsValue
        self.confAttentionLogitCap = confAttentionLogitCap
        self.confNumAttentionHeads = confNumAttentionHeads
        self.confNumHiddenLayers = confNumHiddenLayers
        self.confConvKernelSize = confConvKernelSize
        self.confPositionalBiasSize = confPositionalBiasSize
        self.confReductionFactor = confReductionFactor
        self.confResidualWeight = confResidualWeight
        self.sscpConvChannelSize = sscpConvChannelSize
        self.sscpConvGroupNormEps = sscpConvGroupNormEps
        self.sscpConvKernelSize = sscpConvKernelSize
        self.sscpConvStrideSize = sscpConvStrideSize
        self.vocabSize = vocabSize
        self.sscpConvEps = sscpConvEps
        self.rmsNormEps = rmsNormEps
        self.gradientClipping = gradientClipping
        self.vocabOffset = vocabOffset
    }
    
    enum CodingKeys: String, CodingKey {
        case inputFeatSize = "input_feat_size"
        case hiddenSize = "hidden_size"
        case confAttentionChunkSize = "conf_attention_chunk_size"
        case confAttentionContextLeft = "conf_attention_context_left"
        case confAttentionContextRight = "conf_attention_context_right"
        case confAttentionInvalidLogitsValue = "conf_attention_invalid_logits_value"
        case confAttentionLogitCap = "conf_attention_logit_cap"
        case confNumAttentionHeads = "conf_num_attention_heads"
        case confNumHiddenLayers = "conf_num_hidden_layers"
        case confConvKernelSize = "conf_conv_kernel_size"
        case confPositionalBiasSize = "conf_positional_bias_size"
        case confReductionFactor = "conf_reduction_factor"
        case confResidualWeight = "conf_residual_weight"
        case sscpConvChannelSize = "sscp_conv_channel_size"
        case sscpConvGroupNormEps = "sscp_conv_group_norm_eps"
        case sscpConvKernelSize = "sscp_conv_kernel_size"
        case sscpConvStrideSize = "sscp_conv_stride_size"
        case vocabSize = "vocab_size"
        case sscpConvEps = "sscp_conv_eps"
        case rmsNormEps = "rms_norm_eps"
        case gradientClipping = "gradient_clipping"
        case vocabOffset = "vocab_offset"
    }
}

// MARK: - Vision Configuration

public struct Gemma3nVisionConfiguration: Codable, Sendable {
    public let modelType: String
    public let numHiddenLayers: Int?
    public let hiddenSize: Int
    public let intermediateSize: Int?
    public let numAttentionHeads: Int?
    public let patchSize: Int?
    public let imageSize: Int?
    public let numChannels: Int?
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let vocabOffset: Int
    
    public init(
        modelType: String = "gemma3n_vision",
        numHiddenLayers: Int? = 12,
        hiddenSize: Int = 2048,
        intermediateSize: Int? = 8192,
        numAttentionHeads: Int? = 16,
        patchSize: Int? = 16,
        imageSize: Int? = 224,
        numChannels: Int? = 3,
        rmsNormEps: Float = 1e-6,
        vocabSize: Int = 128,
        vocabOffset: Int = 262_144
    ) {
        self.modelType = modelType
        self.numHiddenLayers = numHiddenLayers
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.patchSize = patchSize
        self.imageSize = imageSize
        self.numChannels = numChannels
        self.rmsNormEps = rmsNormEps
        self.vocabSize = vocabSize
        self.vocabOffset = vocabOffset
    }
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numHiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
        case imageSize = "image_size"
        case numChannels = "num_channels"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabOffset = "vocab_offset"
    }
}

// MARK: - Text Configuration

public struct Gemma3nTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: [Int]
    public let numAttentionHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let vocabSizePerLayerInput: Int
    public let numKeyValueHeads: Int
    public let laurelRank: Int
    public let fracSharedLayers: Float?
    public let altupActiveIdx: Int
    public let padTokenId: Int
    public let altupNumInputs: Int
    public let altupCoefClip: Float?
    public let altupCorrectScale: Bool
    public let hiddenSizePerLayerInput: Int
    public let ropeLocalBaseFreq: Float
    public let ropeTraditional: Bool?
    public let ropeTheta: Float
    public let queryPreAttnScalar: Float?
    public let slidingWindow: Int
    public let ropeScaling: [Float]?
    public let mmTokensPerImage: Int?
    public let slidingWindowPattern: Int?
    public let activationSparsityPattern: [Float]?
    public let finalLogitSoftcapping: Float
    public let queryRescaleScalar: Float?
    public let numKvSharedLayers: Int
    public let maxPositionEmbeddings: Int
    public let attnLogitSoftcapping: Float?
    public let layerTypes: [String]
    
    public init(
        modelType: String,
        hiddenSize: Int,
        numHiddenLayers: Int,
        intermediateSize: [Int],
        numAttentionHeads: Int = 2,
        headDim: Int = 256,
        rmsNormEps: Float = 1.0e-6,
        vocabSize: Int = 262400,
        vocabSizePerLayerInput: Int = 262144,
        numKeyValueHeads: Int = 4,
        laurelRank: Int = 64,
        fracSharedLayers: Float? = 0.5,
        altupActiveIdx: Int = 0,
        padTokenId: Int = 0,
        altupNumInputs: Int = 4,
        altupCoefClip: Float? = nil,
        altupCorrectScale: Bool = true,
        hiddenSizePerLayerInput: Int = 1024,
        ropeLocalBaseFreq: Float = 10000.0,
        ropeTraditional: Bool? = false,
        ropeTheta: Float = 1000000.0,
        queryPreAttnScalar: Float? = 0.0625,
        slidingWindow: Int = 1024,
        ropeScaling: [Float]? = nil,
        mmTokensPerImage: Int? = 256,
        slidingWindowPattern: Int? = 5,
        activationSparsityPattern: [Float]? = nil,
        finalLogitSoftcapping: Float = 30.0,
        queryRescaleScalar: Float? = 1.0,
        numKvSharedLayers: Int = 0,
        maxPositionEmbeddings: Int = 32768,
        attnLogitSoftcapping: Float? = 0.0,
        layerTypes: [String] = []
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.vocabSize = vocabSize
        self.vocabSizePerLayerInput = vocabSizePerLayerInput
        self.numKeyValueHeads = numKeyValueHeads
        self.laurelRank = laurelRank
        self.fracSharedLayers = fracSharedLayers
        self.altupActiveIdx = altupActiveIdx
        self.padTokenId = padTokenId
        self.altupNumInputs = altupNumInputs
        self.altupCoefClip = altupCoefClip
        self.altupCorrectScale = altupCorrectScale
        self.hiddenSizePerLayerInput = hiddenSizePerLayerInput
        self.ropeLocalBaseFreq = ropeLocalBaseFreq
        self.ropeTraditional = ropeTraditional
        self.ropeTheta = ropeTheta
        self.queryPreAttnScalar = queryPreAttnScalar
        self.slidingWindow = slidingWindow
        self.ropeScaling = ropeScaling
        self.mmTokensPerImage = mmTokensPerImage 
        self.slidingWindowPattern = slidingWindowPattern
        self.activationSparsityPattern = activationSparsityPattern
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.queryRescaleScalar = queryRescaleScalar
        self.numKvSharedLayers = numKvSharedLayers
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attnLogitSoftcapping = attnLogitSoftcapping
        self.layerTypes = layerTypes
    }
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKeyValueHeads = "num_key_value_heads"
        case laurelRank = "laurel_rank"
        case fracSharedLayers = "frac_shared_layers"
        case altupActiveIdx = "altup_active_idx"
        case padTokenId = "pad_token_id"
        case altupNumInputs = "altup_num_inputs"
        case altupCoefClip = "altup_coef_clip"
        case altupCorrectScale = "altup_correct_scale"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case ropeTheta = "rope_theta"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case ropeScaling = "rope_scaling"
        case mmTokensPerImage = "mm_tokens_per_image"
        case slidingWindowPattern = "sliding_window_pattern"
        case activationSparsityPattern = "activation_sparsity_pattern"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case queryRescaleScalar = "query_rescale_scalar"
        case numKvSharedLayers = "num_kv_shared_layers"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attnLogitSoftcapping = "attn_logit_softcapping"
        case layerTypes = "layer_types"
    }
}

// MARK: - Model Configuration

public struct Gemma3nConfiguration: Codable, Sendable {
    public let textConfig: Gemma3nTextConfiguration
    public let visionConfig: Gemma3nVisionConfiguration
    public let audioConfig: Gemma3nAudioConfiguration
    public let modelType: String
    public let vocabSize: Int?
    public let ignoreIndex: Int?
    public let imageTokenIndex: Int?
    public let audioTokenId: Int
    public let imageTokenId: Int
    public let hiddenSize: Int?
    public let padTokenId: Int?
    public let visionSoftTokensPerImage: Int
    public let audioSoftTokensPerImage: Int
    public let eosTokenId: [Int]?
    
    public init(
        textConfig: Gemma3nTextConfiguration,
        visionConfig: Gemma3nVisionConfiguration,
        audioConfig: Gemma3nAudioConfiguration,
        modelType: String,
        vocabSize: Int? = 257152,
        ignoreIndex: Int? = -100,
        imageTokenIndex: Int? = 262145,
        audioTokenId: Int = 262273,
        imageTokenId: Int = 262145,
        hiddenSize: Int? = 2048,
        padTokenId: Int? = 0,
        visionSoftTokensPerImage: Int = 256,
        audioSoftTokensPerImage: Int = 188,
        eosTokenId: [Int]? = nil
    ) {
        self.textConfig = textConfig
        self.visionConfig = visionConfig
        self.audioConfig = audioConfig
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.ignoreIndex = ignoreIndex
        self.imageTokenIndex = imageTokenIndex
        self.audioTokenId = audioTokenId
        self.imageTokenId = imageTokenId
        self.hiddenSize = hiddenSize
        self.padTokenId = padTokenId
        self.visionSoftTokensPerImage = visionSoftTokensPerImage
        self.audioSoftTokensPerImage = audioSoftTokensPerImage
        self.eosTokenId = eosTokenId
    }
    
    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case ignoreIndex = "ignore_index"
        case imageTokenIndex = "image_token_index"
        case audioTokenId = "audio_token_id"
        case imageTokenId = "image_token_id"
        case hiddenSize = "hidden_size"
        case padTokenId = "pad_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case audioSoftTokensPerImage = "audio_soft_tokens_per_image"
        case eosTokenId = "eos_token_id"
    }
}