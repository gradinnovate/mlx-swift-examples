// Copyright © 2024 Apple Inc.

import CoreImage
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// Based on https://github.com/ml-explore/mlx-vlm/tree/main/mlx_vlm/models/gemma3n/gemma3n.py

// MARK: - Masked Scatter Helper

private func maskedScatter(
    _ inputTensor: MLXArray,
    _ mask: MLXArray,
    _ source: MLXArray
) -> MLXArray {
    let maskBool = mask.asType(.bool)
    
    // Early exit if no mask is set
    if !maskBool.any().item() {
        return broadcast(inputTensor, to: mask.shape)
    }
    
    let inputShape = mask.shape
    let resultFlat = broadcast(inputTensor, to: inputShape).flattened()
    let maskFlat = maskBool.flattened()
    let sourceFlat = source.flattened()
    
    // Create selection indices using cumulative sum
    let selectionMask = cumsum(maskFlat.asType(.int32), axis: 0) - 1
    
    // Bound check and create source selection
    let sourceLen = sourceFlat.shape[0]
    let boundedIndices = selectionMask % sourceLen
    
    // Vectorized selection from source
    let selectedValues = sourceFlat[boundedIndices]
    
    let result = MLX.where(maskFlat, selectedValues, resultFlat)
    return result.reshaped(inputShape)
}

// MARK: - Multimodal Embedder

private class Gemma3nMultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "hard_embedding_norm") var hardEmbeddingNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "soft_embedding_norm") var softEmbeddingNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    let embeddingPostProjectionNorm: Gemma3nRMSNormNoWeight
    
    let multimodalHiddenSize: Int
    let eps: Float
    let vocabOffset: Int
    let vocabSize: Int
    let textHiddenSize: Int
    
    init<T>(multimodalConfig: T, textConfig: Gemma3nTextConfiguration) {
        // Extract properties based on config type
        if let visionConfig = multimodalConfig as? Gemma3nVisionConfiguration {
            self.multimodalHiddenSize = visionConfig.hiddenSize
            self.vocabOffset = visionConfig.vocabOffset
            self.vocabSize = visionConfig.vocabSize
        } else if let audioConfig = multimodalConfig as? Gemma3nAudioConfiguration {
            self.multimodalHiddenSize = audioConfig.hiddenSize
            self.vocabOffset = audioConfig.vocabOffset
            self.vocabSize = audioConfig.vocabSize
        } else {
            fatalError("Unsupported multimodalConfig type in Gemma3nMultimodalEmbedder initializer")
        }
        
        self.eps = 1e-6  // Default RMS norm eps
        self.textHiddenSize = textConfig.hiddenSize
        
        self._embedding.wrappedValue = Embedding(
            embeddingCount: vocabSize,
            dimensions: multimodalHiddenSize
        )
        self._hardEmbeddingNorm.wrappedValue = Gemma3nRMSNorm(
            dimensions: multimodalHiddenSize,
            eps: eps
        )
        self._softEmbeddingNorm.wrappedValue = Gemma3nRMSNorm(
            dimensions: multimodalHiddenSize,
            eps: eps
        )
        self._embeddingProjection.wrappedValue = Linear(
            multimodalHiddenSize, textHiddenSize, bias: false
        )
        self.embeddingPostProjectionNorm = Gemma3nRMSNormNoWeight(
            dimensions: textHiddenSize,
            eps: eps
        )
    }
    
    func callAsFunction(
        inputIds: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil
    ) -> MLXArray {
        // Ensure exactly one of inputIds or inputsEmbeds is provided
        guard (inputIds == nil) != (inputsEmbeds == nil) else {
            fatalError("You must specify exactly one of inputIds or inputsEmbeds")
        }
        
        let embNorm: MLXArray
        if let inputsEmbeds = inputsEmbeds {
            embNorm = softEmbeddingNorm(inputsEmbeds)
        } else {
            let hardEmb = embedding(inputIds! - vocabOffset)
            embNorm = hardEmbeddingNorm(hardEmb)
        }
        
        let embNormProj = embeddingProjection(embNorm)
        let projected = embeddingPostProjectionNorm(embNormProj)
        return projected
    }
}

// MARK: - Gemma3n Model

public class Gemma3n: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: Gemma3nLanguageModel
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma3nVisionModel
    @ModuleInfo(key: "audio_tower") private var audioTower: Gemma3nAudioModel
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma3nMultimodalEmbedder
    @ModuleInfo(key: "embed_audio") private var embedAudio: Gemma3nMultimodalEmbedder
    
    public let config: Gemma3nConfiguration
    public let modelType: String
    
    public var vocabularySize: Int { config.textConfig.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    
    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        return languageModel.newCache(parameters: parameters)
    }
    
    public init(_ config: Gemma3nConfiguration) {
        self.config = config
        self.modelType = config.modelType
        
        self._languageModel.wrappedValue = Gemma3nLanguageModel(config.textConfig)
        self._visionTower.wrappedValue = Gemma3nVisionModel(config: config.visionConfig)
        self._audioTower.wrappedValue = Gemma3nAudioModel(config: config.audioConfig)
        
        self._embedVision.wrappedValue = Gemma3nMultimodalEmbedder(
            multimodalConfig: config.visionConfig,
            textConfig: config.textConfig
        )
        self._embedAudio.wrappedValue = Gemma3nMultimodalEmbedder(
            multimodalConfig: config.audioConfig,
            textConfig: config.textConfig
        )
    }
    
    private func getInputEmbeddings(
        inputIds: MLXArray? = nil,
        pixelValues: MLXArray? = nil,
        inputFeatures: MLXArray? = nil,
        inputFeaturesMask: MLXArray? = nil
    ) -> (MLXArray, MLXArray?) {
        var inputsEmbeds = languageModel.model.embedTokens(inputIds!)
        
        let perLayerInputsMask = logicalAnd(
            greaterEqual(inputIds!, MLXArray(0)),
            less(inputIds!, MLXArray(config.textConfig.vocabSizePerLayerInput))
        )
        let perLayerInputsTokens = MLX.where(
            perLayerInputsMask,
            inputIds!,
            zeros(like: inputIds!)
        )
        let perLayerInputs = languageModel.model.getPerLayerInputs(perLayerInputsTokens)
        
        if pixelValues == nil && inputFeatures == nil {
            return (inputsEmbeds, perLayerInputs)
        }
        
        if let inputIds = inputIds {
            // Handle vision tokens
            let visionMask = logicalAnd(
                greaterEqual(inputIds, MLXArray(embedVision.vocabOffset)),
                less(inputIds, MLXArray(embedAudio.vocabOffset))
            )
            let dummyVisionTokenId = embedVision.vocabOffset + embedVision.vocabSize - 1
            let visionTokens = MLX.where(visionMask, inputIds, MLXArray(dummyVisionTokenId))
            let visionEmbedsFlat = embedVision(inputIds: visionTokens)
            inputsEmbeds = MLX.where(
                visionMask.expandedDimensions(axis: -1),
                visionEmbedsFlat,
                inputsEmbeds
            )
            
            // Handle audio tokens
            let audioMask = greaterEqual(inputIds, MLXArray(embedAudio.vocabOffset))
            let dummyAudioTokenId = embedAudio.vocabOffset + embedAudio.vocabSize - 1
            let audioTokens = MLX.where(audioMask, inputIds, MLXArray(dummyAudioTokenId))
            let audioEmbedsFlat = embedAudio(inputIds: audioTokens)
            inputsEmbeds = MLX.where(
                audioMask.expandedDimensions(axis: -1),
                audioEmbedsFlat,
                inputsEmbeds
            )
        }
        
        // Vision features
        if let pixelValues = pixelValues {
            let imageFeatures = Self.getImageFeatures(
                pixelValues: pixelValues,
                visionTower: visionTower,
                config: config,
                embedVision: embedVision
            )
            
            let modality = "image"
            inputsEmbeds = Self.mergeMultimodalAndText(
                inputsEmbeds,
                imageFeatures,
                constructSpecialModalityMask(
                    inputIds: inputIds,
                    inputsEmbeds: inputsEmbeds,
                    tokenId: config.imageTokenId,
                    modality: modality
                ),
                modality: modality
            )
        }
        
        // Audio features
        if let inputFeatures = inputFeatures, let inputFeaturesMask = inputFeaturesMask {
            let (audioFeatures, audioMask) = getAudioFeatures(
                inputFeatures,
                logicalNot(inputFeaturesMask)
            )
            let vocabSize = config.vocabSize ?? 257152
            let audioPaddingIds = MLXArray([vocabSize - 1])
            let audioPaddingEmbs = embedAudio(inputIds: audioPaddingIds)
            let maskedAudioFeatures = MLX.where(
                audioMask.expandedDimensions(axis: -1),
                audioPaddingEmbs,
                audioFeatures
            )
            
            let (audioBatchSize, audioSeqLen, audioEmbedDim) = (
                maskedAudioFeatures.shape[0],
                maskedAudioFeatures.shape[1],
                maskedAudioFeatures.shape[2]
            )
            let extraPaddingTokens = config.audioSoftTokensPerImage - audioSeqLen
            let extraPaddingFeatures = broadcast(
                audioPaddingEmbs,
                to: [audioBatchSize, extraPaddingTokens, audioEmbedDim]
            )
            
            let finalAudioFeatures = concatenated([maskedAudioFeatures, extraPaddingFeatures], axis: 1)
            let modality = "audio"
            inputsEmbeds = Self.mergeMultimodalAndText(
                inputsEmbeds,
                finalAudioFeatures,
                constructSpecialModalityMask(
                    inputIds: inputIds,
                    inputsEmbeds: inputsEmbeds,
                    tokenId: config.audioTokenId,
                    modality: modality
                ),
                modality: modality
            )
        }
        
        return (inputsEmbeds, perLayerInputs)
    }
    
    private func getAudioFeatures(
        _ inputFeatures: MLXArray,
        _ inputFeaturesMask: MLXArray
    ) -> (MLXArray, MLXArray) {
        let (audioOutputs, audioMask) = audioTower(inputFeatures, inputFeaturesMask)
        return (embedAudio(inputsEmbeds: audioOutputs), audioMask)
    }
    
    private static func getImageFeatures(
        pixelValues: MLXArray,
        visionTower: Gemma3nVisionModel,
        config: Gemma3nConfiguration,
        embedVision: Gemma3nMultimodalEmbedder
    ) -> MLXArray {
        var visionOutputs = visionTower(pixelValues, outputHiddenStates: true)
        visionOutputs = visionOutputs.transposed(0, 3, 1, 2)
        visionOutputs = visionOutputs.reshaped([
            visionOutputs.shape[0],
            config.visionConfig.hiddenSize,
            config.visionSoftTokensPerImage
        ]).transposed(0, 2, 1)
        
        // Normalize and embed the soft tokens into language model space
        visionOutputs = visionOutputs * sqrt(Float(config.visionConfig.hiddenSize))
        return embedVision(inputsEmbeds: visionOutputs)
    }
    
    private func constructSpecialModalityMask(
        inputIds: MLXArray?,
        inputsEmbeds: MLXArray,
        tokenId: Int,
        modality: String = "image"
    ) -> MLXArray {
        if let inputIds = inputIds {
            let specialModalityMask = equal(inputIds, MLXArray(tokenId)).expandedDimensions(axis: -1)
            return broadcast(specialModalityMask, to: inputsEmbeds.shape)
        } else {
            let targetEmbed: MLXArray
            if modality == "audio" {
                targetEmbed = embedAudio(inputIds: MLXArray([tokenId]))
            } else {
                targetEmbed = languageModel.model.embedTokens(MLXArray([tokenId]))
            }
            return equal(inputsEmbeds, targetEmbed)
        }
    }
    
    private static func mergeMultimodalAndText(
        _ inputsEmbeds: MLXArray,
        _ features: MLXArray,
        _ specialModalityMask: MLXArray,
        modality: String = "image"
    ) -> MLXArray {
        // Count special tokens by summing the mask
        let modalityTokensInText = specialModalityMask.sum()
        let featureTokens = features.size
        
        if modalityTokensInText.item(Int.self) != featureTokens {
            fatalError(
                """
                Number of \(modality)s does not match number of special \(modality) tokens in the input text.
                Got \(modalityTokensInText.item(Int.self)) \(modality) tokens in the text and
                \(featureTokens) tokens from \(modality) embeddings.
                """
            )
        }
        
        let featuresTyped = features.asType(inputsEmbeds.dtype)
        return maskedScatter(inputsEmbeds, specialModalityMask, featuresTyped)
    }
    
    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws -> PrepareResult {
        // Handle text-only input
        if input.image?.pixels == nil {
            let convertedCache = cache.compactMap { $0 as? KVCache }
            let result = languageModel(
                input.text.tokens,
                inputsEmbeds: nil,
                mask: nil,
                cache: convertedCache,
                perLayerInputs: nil
            )
            return .logits(result)
        }
        
        // Handle multimodal input
        let (inputEmbeddings, perLayerInputs) = getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: input.image?.pixels,
            inputFeatures: nil,
            inputFeaturesMask: nil
        )
        
        let convertedCache = cache.compactMap { $0 as? KVCache }
        let result = languageModel(
            nil,  // Pass nil for tokens when using embeddings
            inputsEmbeds: inputEmbeddings,
            mask: nil,
            cache: convertedCache,
            perLayerInputs: perLayerInputs
        )
        
        return .logits(result)
    }
    
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let convertedCache = cache?.compactMap { $0 as? KVCache }
        return languageModel(inputs, cache: convertedCache).logits
    }
    
    public func callAsFunction(
        _ inputIds: MLXArray,
        pixelValues: MLXArray,
        mask: MLXArray? = nil,
        cache: [any KVCache]? = nil,
        inputFeatures: MLXArray? = nil,
        inputFeaturesMask: MLXArray? = nil
    ) -> MLXArray {
        let (inputsEmbeds, perLayerInputs) = getInputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelValues,
            inputFeatures: inputFeatures,
            inputFeaturesMask: inputFeaturesMask
        )
        
        let convertedCache = cache?.compactMap { $0 as? KVCache }
        let logits = languageModel(
            nil,
            inputsEmbeds: inputsEmbeds,
            mask: mask,
            cache: convertedCache,
            perLayerInputs: perLayerInputs
        )
        return logits.logits
    }
    
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]
        
        for (k, v) in weights {
            if k.starts(with: "model.") {
                let newKey = String(k.dropFirst(6))  // Remove "model." prefix
                sanitizedWeights[newKey] = v
            } else {
                sanitizedWeights[k] = v
            }
        }
        
        // Apply language model sanitization
        sanitizedWeights = languageModel.sanitize(weights: sanitizedWeights)
        
        // Apply vision tower sanitization
        sanitizedWeights = visionTower.sanitize(weights: sanitizedWeights)
        
        // Apply audio tower sanitization  
        sanitizedWeights = audioTower.sanitize(weights: sanitizedWeights)
        
        return sanitizedWeights
    }
    
    public var layers: [Module] {
        return languageModel.model.layers.map { $0 as Module }
    }
}

// MARK: - Processor

public class Gemma3nProcessor: UserInputProcessor {
    private let config: Gemma3nProcessorConfiguration
    private let tokenizer: any Tokenizer
    
    public init(_ config: Gemma3nProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }
    
    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (MLXArray, THW) {
        let userProcessing = processing ?? UserInput.Processing()
        let targetSize = CGSize(width: config.imageSize, height: config.imageSize)
        
        let processedImages = try images.map { image in
            let processedImage = MediaProcessing.apply(image, processing: userProcessing)
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
            let resizedImage = MediaProcessing.resampleBicubic(srgbImage, to: targetSize)
            let normalizedImage = MediaProcessing.normalize(
                resizedImage,
                mean: config.imageMeanTuple,
                std: config.imageStdTuple
            )
            return MediaProcessing.asMLXArray(normalizedImage)
        }
        
        let pixelValues = concatenated(processedImages)
        return (pixelValues, THW(images.count, config.imageSize, config.imageSize))
    }
    
    public func prepare(input: UserInput) async throws -> LMInput {
        // Use structured content message generator
        let messages = Qwen2VLMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(messages: messages)
        
        var processedImage: LMInput.ProcessedImage?
        
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let imagePixelsConcatenated = concatenated(imagePixelsAndFrames.map { $0.0 })
            processedImage = LMInput.ProcessedImage(
                pixels: imagePixelsConcatenated,
                frames: imagePixelsAndFrames.map { $0.1 }
            )
            
            // Expand image tokens
            let startOfImageTokenId = 255999
            let imageTokenId = config.imageTokenId
            let numImageTokens = config.imageSeqLength
            
            var expandedTokens: [Int] = []
            for token in promptTokens {
                if token == startOfImageTokenId {
                    expandedTokens.append(contentsOf: Array(repeating: imageTokenId, count: numImageTokens))
                } else {
                    expandedTokens.append(token)
                }
            }
            promptTokens = expandedTokens
        }
        
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage
        )
    }
}

// MARK: - Processor Configuration

public struct Gemma3nProcessorConfiguration: Codable, Sendable {
    public let processorClass: String
    public let imageProcessorType: String
    public let doNormalize: Bool
    public let doRescale: Bool
    public let doResize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let resample: Int
    public let rescaleFactor: Float
    public let size: ImageSize
    public let doConvertRgb: Bool?
    public let doPanAndScan: Bool?
    public let panAndScanMaxNumCrops: Int?
    public let panAndScanMinCropSize: Int?
    public let panAndScanMinRatioToActivate: Float?
    public let imageTokenId: Int
    
    public struct ImageSize: Codable, Sendable {
        public let height: Int
        public let width: Int
        
        public init(height: Int, width: Int) {
            self.height = height
            self.width = width
        }
    }
    
    public var imageSize: Int { size.height }
    
    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }
    
    public init(
        processorClass: String = "Gemma3nProcessor",
        imageProcessorType: String = "Gemma3nImageProcessor",
        doNormalize: Bool = true,
        doRescale: Bool = true,
        doResize: Bool = true,
        imageMean: [CGFloat] = [0.485, 0.456, 0.406],
        imageStd: [CGFloat] = [0.229, 0.224, 0.225],
        imageSeqLength: Int = 256,
        resample: Int = 2,
        rescaleFactor: Float = 0.00392156862745098,
        size: ImageSize = ImageSize(height: 224, width: 224),
        doConvertRgb: Bool? = true,
        doPanAndScan: Bool? = false,
        panAndScanMaxNumCrops: Int? = nil,
        panAndScanMinCropSize: Int? = nil,
        panAndScanMinRatioToActivate: Float? = nil,
        imageTokenId: Int = 262145
    ) {
        self.processorClass = processorClass
        self.imageProcessorType = imageProcessorType
        self.doNormalize = doNormalize
        self.doRescale = doRescale
        self.doResize = doResize
        self.imageMean = imageMean
        self.imageStd = imageStd
        self.imageSeqLength = imageSeqLength
        self.resample = resample
        self.rescaleFactor = rescaleFactor
        self.size = size
        self.doConvertRgb = doConvertRgb
        self.doPanAndScan = doPanAndScan
        self.panAndScanMaxNumCrops = panAndScanMaxNumCrops
        self.panAndScanMinCropSize = panAndScanMinCropSize
        self.panAndScanMinRatioToActivate = panAndScanMinRatioToActivate
        self.imageTokenId = imageTokenId
    }
    
    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessorType = "image_processor_type"
        case doNormalize = "do_normalize"
        case doRescale = "do_rescale"
        case doResize = "do_resize"
        case doConvertRgb = "do_convert_rgb"
        case doPanAndScan = "do_pan_and_scan"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case resample
        case rescaleFactor = "rescale_factor"
        case size
        case panAndScanMaxNumCrops = "pan_and_scan_max_num_crops"
        case panAndScanMinCropSize = "pan_and_scan_min_crop_size"
        case panAndScanMinRatioToActivate = "pan_and_scan_min_ratio_to_activate"
        case imageTokenId = "image_token_id"
    }
}

// MARK: - LoRA Support

extension Gemma3n: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        return languageModel.model.layers.map { ($0.selfAttn, ["q_proj", "v_proj"]) }
    }
}