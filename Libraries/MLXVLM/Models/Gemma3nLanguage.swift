// Copyright © 2024 Apple Inc.

import CoreImage
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// Based on https://github.com/ml-explore/mlx-vlm/tree/main/mlx_vlm/models/gemma3n/language.py

// MARK: - RMS Normalization

public class Gemma3nRMSNorm: Module {
    let dimensions: Int
    let eps: Float
    let scaleShift: Float
    
    @ParameterInfo(key: "weight") var weight: MLXArray
    
    public init(
        dimensions: Int,
        eps: Float = 1e-6,
        scaleShift: Float = 0.0
    ) {
        self.dimensions = dimensions
        self.eps = eps
        self.scaleShift = scaleShift        
        self._weight.wrappedValue = ones([dimensions])
        super.init()
    }
    
    private func norm(_ x: MLXArray) -> MLXArray {
        return x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = norm(x.asType(.float32))
        
        output = output * (weight + scaleShift)
        
        return output.asType(x.dtype)
    }
}

public class Gemma3nRMSNormNoWeight: Module {
    let dimensions: Int
    let eps: Float
  

    public init(
        dimensions: Int,
        eps: Float = 1e-6
    ) {
        self.dimensions = dimensions
        self.eps = eps
        super.init()
    }
    
    private func norm(_ x: MLXArray) -> MLXArray {
        return x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let output = norm(x.asType(.float32))
    
        return output.asType(x.dtype)
    }
}

private class RMSNoScale: Module {
    let eps: Float
    
    init(eps: Float = 1e-5) {
        self.eps = eps
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)]), eps: eps)
    }
}

// MARK: - Laurel Block

public class Gemma3nLaurelBlock: Module {
    @ModuleInfo(key: "linear_left") var linearLeft: Linear
    @ModuleInfo(key: "linear_right") var linearRight: Linear
    @ModuleInfo(key: "post_laurel_norm") var postLaurelNorm: RMSNorm
    
    let config: Gemma3nTextConfiguration
    
    init(config: Gemma3nTextConfiguration) {
        self.config = config
        
        self._linearLeft.wrappedValue = Linear(
            config.hiddenSize, config.laurelRank, bias: false
        )
        self._linearRight.wrappedValue = Linear(
            config.laurelRank, config.hiddenSize, bias: false
        )
        self._postLaurelNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let laurelX = linearRight(linearLeft(x))
        let normedLaurelX = postLaurelNorm(laurelX)
        return x + normedLaurelX
    }
}

// MARK: - Attention

public class Gemma3nAttention: Module {
    let isSliding: Bool
    let numHeads: Int
    let numKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isKVSharedLayer: Bool
    
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    fileprivate let vNorm: RMSNoScale
    
    let rope: RoPE
    
    init(config: Gemma3nTextConfiguration, layerIdx: Int, isKVSharedLayer: Bool) {
        self.layerIdx = layerIdx
        self.isSliding = config.layerTypes[layerIdx] == "sliding_attention"
        
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.repeats = numHeads / numKVHeads
        self.headDim = config.headDim
        
        self.scale = 1.0
        self.isKVSharedLayer = isKVSharedLayer
        
        self._qProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)
        
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self.vNorm = RMSNoScale(eps: config.rmsNormEps)
        
        let baseFreq = isSliding ? config.ropeLocalBaseFreq : config.ropeTheta
        self.rope = RoPE(
            dimensions: headDim,
            traditional: config.ropeTraditional ?? false,
            base: baseFreq
        )
        super.init()
    }
    
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        
        var queries = qProj(x).reshaped(B, L, -1, headDim)
        queries = qNorm(queries)
        
        var offset = 0
        var keys: MLXArray
        var values: MLXArray
        
        if isKVSharedLayer && cache != nil {
            // For shared layers, retrieve KV from the designated cache layer
            let state = cache!.state
            keys = state[0]
            values = state[1]
            offset = cache!.offset
        } else {
            if let cache = cache {
                offset = cache.offset
            }
            
            keys = kProj(x).reshaped(B, L, -1, headDim)
            keys = kNorm(keys)
            keys = keys.transposed(0, 2, 1, 3)
            keys = rope(keys, offset: offset)
            
            values = vProj(x).reshaped(B, L, -1, headDim)
            values = vNorm(values)
            values = values.transposed(0, 2, 1, 3)
            
            if let cache = cache {
                let updated = cache.update(keys: keys, values: values)
                keys = updated.0
                values = updated.1
            }
        }
        
        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)
        
        var finalMask = mask
        if let maskArray = mask, maskArray.shape.last! != keys.shape[2] {
            let keyLen = keys.shape[2]
            let slicedMask = maskArray[.ellipsis, (-keyLen)...]
            finalMask = slicedMask
        }
        
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: finalMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        
        return oProj(output)
    }
}

// Note: Using MLXFast.scaledDotProductAttention for optimized Metal GPU performance
// This provides:
// - Optimized Metal kernels for query sequence length = 1 (inference)
// - Automatic float32 softmax for numerical stability  
// - Native support for Multi-Query and Grouped Query Attention
// - Efficient memory usage without pre-tiling keys/values

// MARK: - MLP with TopK GELU

public func geluTopK(_ inputs: MLXArray, _ stdMultiplier: MLXArray) -> MLXArray {
    let inputsMean = inputs.mean(axis: -1, keepDims: true)
    let inputsStd = std(inputs, axis: -1, keepDims: true, ddof: 0)
    let cutoffX = inputsMean + inputsStd * stdMultiplier.asType(inputsStd.dtype)
    return geluApproximate(maximum(MLXArray(0), inputs - cutoffX))
}

public class Gemma3nMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    
    let config: Gemma3nTextConfiguration
    let activationSparsity: Float
    let _stdMultiplier: MLXArray?
    
    init(config: Gemma3nTextConfiguration, layerIdx: Int = 0) {
        self.config = config
        
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize[0]
        
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        
        if let activationSparsityPattern = config.activationSparsityPattern {
            self.activationSparsity = activationSparsityPattern[layerIdx]
        } else {
            self.activationSparsity = 0.0
        }
        
        if activationSparsity > 0 {
            let value = sqrt(2.0) * erfInverse(MLXArray(2 * activationSparsity - 1))
            self._stdMultiplier = value
        } else {
            self._stdMultiplier = nil
        }
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateProj = self.gateProj(x)
        let activations: MLXArray
        
        if activationSparsity > 0.0, let stdMultiplier = _stdMultiplier {
            activations = geluTopK(gateProj, stdMultiplier)
        } else {
            activations = geluApproximate(gateProj)
        }
        
        let upProj = self.upProj(x)
        let downProj = self.downProj(activations * upProj)
        return downProj
    }
}

// MARK: - AltUp (Alternating Updates)

public class Gemma3nAltUp: Module {
    @ParameterInfo(key: "correct_output_scale") var correctOutputScale: MLXArray
    @ModuleInfo(key: "correction_coefs") var correctionCoefs: Linear
    @ModuleInfo(key: "prediction_coefs") var predictionCoefs: Linear
    @ModuleInfo(key: "modality_router") var modalityRouter: Linear
    @ModuleInfo(key: "router_norm") var routerNorm: RMSNorm
    
    let config: Gemma3nTextConfiguration
    
    init(config: Gemma3nTextConfiguration) {
        self.config = config
        
        self._correctOutputScale.wrappedValue = zeros([config.hiddenSize])
        
        self._correctionCoefs.wrappedValue = Linear(
            config.altupNumInputs, config.altupNumInputs, bias: false
        )
        self._predictionCoefs.wrappedValue = Linear(
            config.altupNumInputs, config.altupNumInputs * config.altupNumInputs, bias: false
        )
        self._modalityRouter.wrappedValue = Linear(
            config.hiddenSize, config.altupNumInputs, bias: false
        )
        self._routerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }
    
    func computeRouterModalities(_ x: MLXArray) -> MLXArray {
        let routerInputs = routerNorm(x) * pow(Float(config.hiddenSize), -1.0)
        let routed = modalityRouter(routerInputs).asType(.float32)
        return tanh(routed)
    }
    
    func predict(_ x: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(x[config.altupActiveIdx])
        
        var predictionCoefsWeight = predictionCoefs.weight.asType(.float32)
        
        if let coefClip = config.altupCoefClip {
            predictionCoefsWeight = clip(
                predictionCoefsWeight,
                min: -coefClip, max: coefClip
            )
        }
        
        // Apply prediction coefficients
        let allCoefs = matmul(modalities, predictionCoefsWeight)
            .reshaped(
                modalities.shape[0], modalities.shape[1],
                config.altupNumInputs, config.altupNumInputs
            )
            .transposed(0, 1, 3, 2)
        
        let xUp = x.asType(.float32)
        let xPermuted = xUp.transposed(1, 2, 3, 0)
        let predictions = matmul(xPermuted, allCoefs).transposed(3, 0, 1, 2)
        
        return (predictions + xUp).asType(x.dtype)
    }
    
    func correct(_ predictions: MLXArray, _ activated: MLXArray) -> MLXArray {
        let modalities = computeRouterModalities(activated)
        
        var correctionCoefsWeight = correctionCoefs.weight.asType(.float32)
        
        if let coefClip = config.altupCoefClip {
            correctionCoefsWeight = clip(
                correctionCoefsWeight,
                min: -coefClip, max: coefClip
            )
        }
        
        let allCoefs = matmul(modalities, correctionCoefsWeight) + 1.0
        
        let activeX = predictions[config.altupActiveIdx]
        let innovation = activated - activeX
        
        let allCoefsTransposed = allCoefs.transposed(2, 1, 0)
        let corrected = innovation.expandedDimensions(axis: 0) * allCoefsTransposed[0..., .newAxis, 0...]
        
        return (corrected + predictions).asType(activated.dtype)
    }
}

// MARK: - Decoder Layer

public class Gemma3nDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: Gemma3nAttention
    @ModuleInfo(key: "mlp") public var mlp: Gemma3nMLP
    @ModuleInfo(key: "input_layernorm") public var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") public var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") public var postFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "altup") public var altup: Gemma3nAltUp
    @ModuleInfo(key: "laurel") public var laurel: Gemma3nLaurelBlock
    @ModuleInfo(key: "per_layer_input_gate") public var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") public var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") public var postPerLayerInputNorm: RMSNorm
    
    let config: Gemma3nTextConfiguration
    let hiddenSize: Int
    let layerIdx: Int
    let isSliding: Bool
    let slidingWindow: Int
    let hiddenSizePerLayerInput: Int
    
    init(config: Gemma3nTextConfiguration, layerIdx: Int, isKVSharedLayer: Bool) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.layerIdx = layerIdx
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
        
        self._selfAttn.wrappedValue = Gemma3nAttention(
            config: config, layerIdx: layerIdx, isKVSharedLayer: isKVSharedLayer
        )
        self._mlp.wrappedValue = Gemma3nMLP(config: config, layerIdx: layerIdx)
        
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize, eps: config.rmsNormEps
        )
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize, eps: config.rmsNormEps
        )
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize, eps: config.rmsNormEps
        )
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize, eps: config.rmsNormEps
        )
        
        self.isSliding = self._selfAttn.wrappedValue.isSliding
        self.slidingWindow = config.slidingWindow
        self._altup.wrappedValue = Gemma3nAltUp(config: config)
        self._laurel.wrappedValue = Gemma3nLaurelBlock(config: config)
        
        self._perLayerInputGate.wrappedValue = Linear(
            hiddenSize, hiddenSizePerLayerInput, bias: false
        )
        self._perLayerProjection.wrappedValue = Linear(
            hiddenSizePerLayerInput, hiddenSize, bias: false
        )
        self._postPerLayerInputNorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize, eps: config.rmsNormEps
        )
        
        super.init()
    }
    
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        let predictions = altup.predict(x)
        let activePrediction = predictions[config.altupActiveIdx]
        
        let activePredictionNormed = inputLayernorm(activePrediction)
        let laurelOutput = laurel(activePredictionNormed)
        
        let attn = selfAttn(activePredictionNormed, mask: mask, cache: cache)
        let attnNormed = postAttentionLayernorm(attn)
        let attnGated = activePrediction + attnNormed
        let attnLaurel = (attnGated + laurelOutput) * pow(2.0, -0.5)
        
        let attnNorm = preFeedforwardLayernorm(attnLaurel)
        let attnFfw = mlp(attnNorm)
        let attnFfwNorm = postFeedforwardLayernorm(attnFfw)
        let attnFfwLaurelGated = attnLaurel + attnFfwNorm
        
        var correctedPredictions = altup.correct(predictions, attnFfwLaurelGated)
        
        var firstPrediction = correctedPredictions[config.altupActiveIdx]
        if config.altupCorrectScale {
            firstPrediction = firstPrediction * altup.correctOutputScale
        }
        
        firstPrediction = perLayerInputGate(firstPrediction)
        firstPrediction = geluApproximate(firstPrediction)
        
        if let perLayerInput = perLayerInput {
            firstPrediction = firstPrediction * perLayerInput
        }
        
        firstPrediction = perLayerProjection(firstPrediction)
        firstPrediction = postPerLayerInputNorm(firstPrediction)
        
        correctedPredictions[1...] = correctedPredictions[1...] + firstPrediction
        
        return correctedPredictions
    }
}

// MARK: - Gemma3 Model

public class Gemma3Model: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "layers") public var layers: [Gemma3nDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm
    @ModuleInfo(key: "altup_projections") var altupProjections: [Linear]
    @ModuleInfo(key: "altup_unembed_projections") var altupUnembedProjections: [Linear]
    
    let config: Gemma3nTextConfiguration
    let hiddenSize: Int
    let hiddenSizePerLayerInput: Int
    let vocabSize: Int
    let vocabSizePerLayerInput: Int
    let numHiddenLayers: Int
    let firstKVSharedLayerIdx: Int
    let firstSlidingIdx: Int
    let firstFullIdx: Int
    let layerIdxToCacheIdx: [Int]
    
    init(_ config: Gemma3nTextConfiguration) {
        self.config = config
        self.hiddenSize = config.hiddenSize
        self.hiddenSizePerLayerInput = config.hiddenSizePerLayerInput
        self.vocabSize = config.vocabSize
        self.vocabSizePerLayerInput = config.vocabSizePerLayerInput
        self.numHiddenLayers = config.numHiddenLayers
        self.firstKVSharedLayerIdx = numHiddenLayers - config.numKvSharedLayers
        
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: vocabSize,
            dimensions: hiddenSize
        )
        
        // Capture values to avoid self capture in closure
        let localFirstKVSharedLayerIdx = self.firstKVSharedLayerIdx
        self._layers.wrappedValue = (0..<numHiddenLayers).map { layerIdx in
            Gemma3nDecoderLayer(
                config: config,
                layerIdx: layerIdx,
                isKVSharedLayer: layerIdx >= localFirstKVSharedLayerIdx
            )
        }
        
        self._embedTokensPerLayer.wrappedValue = Embedding(
            embeddingCount: vocabSizePerLayerInput,
            dimensions: numHiddenLayers * hiddenSizePerLayerInput
        )
        
        self._perLayerModelProjection.wrappedValue = Linear(
            hiddenSize,
            numHiddenLayers * hiddenSizePerLayerInput,
            bias: false
        )
        
        self._perLayerProjectionNorm.wrappedValue = RMSNorm(
            dimensions: hiddenSizePerLayerInput,
            eps: config.rmsNormEps
        )
        
        // Capture hiddenSize to avoid self capture in closure
        let localHiddenSize = self.hiddenSize
        self._altupProjections.wrappedValue = (1..<config.altupNumInputs).map { _ in
            Linear(localHiddenSize, localHiddenSize, bias: false)
        }
        
        self._altupUnembedProjections.wrappedValue = (1..<config.altupNumInputs).map { _ in
            Linear(localHiddenSize, localHiddenSize, bias: false)
        }
        
        self._norm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: config.rmsNormEps)
        
        self.firstSlidingIdx = config.layerTypes.firstIndex(of: "sliding_attention") ?? 0
        self.firstFullIdx = config.layerTypes.firstIndex(of: "full_attention") ?? 0
        
        // Calculate layer index to cache index mapping
        var layerIdxToCacheIdx: [Int] = []
        let concreteLayerTypes = Array(config.layerTypes[..<firstKVSharedLayerIdx])
        let sharedFullIdx = concreteLayerTypes.lastIndex(of: "full_attention") ?? 0
        let sharedSlidingIdx = concreteLayerTypes.lastIndex(of: "sliding_attention") ?? 0
        
        for (i, layerType) in config.layerTypes.enumerated() {
            if i < firstKVSharedLayerIdx {
                layerIdxToCacheIdx.append(i)
            } else {
                if layerType == "full_attention" {
                    layerIdxToCacheIdx.append(sharedFullIdx)
                } else if layerType == "sliding_attention" {
                    layerIdxToCacheIdx.append(sharedSlidingIdx)
                } else {
                    fatalError("Unknown layer type: \(layerType)")
                }
            }
        }
        self.layerIdxToCacheIdx = layerIdxToCacheIdx
        
        super.init()
    }
    
    public func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        let perLayerInputsMask = less(inputIds, MLXArray(vocabSizePerLayerInput))
        let tokens = MLX.where(perLayerInputsMask, inputIds, zeros(like: inputIds))
        let result = embedTokensPerLayer(tokens) * sqrt(Float(hiddenSizePerLayerInput))
        return result.reshaped(
            inputIds.shape + [numHiddenLayers, hiddenSizePerLayerInput]
        )
    }
    
    func projectPerLayerInputs(
        _ inputsEmbeds: MLXArray,
        _ perLayerInputs: MLXArray
    ) -> MLXArray {
        let perLayerProjection = perLayerModelProjection(inputsEmbeds) * pow(Float(hiddenSize), -0.5)
        let reshaped = perLayerProjection.reshaped(
            inputsEmbeds.shape[0..<inputsEmbeds.ndim-1] + [numHiddenLayers, hiddenSizePerLayerInput]
        )
        let normed = perLayerProjectionNorm(reshaped)
        return (normed + perLayerInputs) * pow(2.0, -0.5)
    }
    
    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        
        if let inputsEmbeds = inputsEmbeds {
            h = inputsEmbeds
        } else if let inputs = inputs {
            h = embedTokens(inputs) * sqrt(Float(hiddenSize))
        } else {
            fatalError("Either inputs or inputsEmbeds must be provided")
        }
        
        var actualPerLayerInputs = perLayerInputs
        if actualPerLayerInputs == nil, let inputs = inputs {
            actualPerLayerInputs = getPerLayerInputs(inputs)
        }
        
        if let perLayerInputsValue = actualPerLayerInputs {
            actualPerLayerInputs = projectPerLayerInputs(h, perLayerInputsValue)
        }
        
        let cache = cache ?? Array(repeating: nil as KVCache?, count: layers.count)
        
        // Create attention masks
        var fullMask: MLXArray? = nil
        var slidingWindowMask: MLXArray? = nil
        
        if mask == nil {
            // Create attention masks for global and sliding window layers
            fullMask = createAttentionMask(h: h, cache: Array(cache[firstFullIdx...].compactMap { $0 }))
            slidingWindowMask = createAttentionMask(h: h, cache: Array(cache[firstSlidingIdx...].compactMap { $0 }))
        }
        
        let h0 = h
        
        // Expand hidden states to support per-layer inputs
        let targetMagnitude = sqrt(h0.square().mean(axis: -1, keepDims: true))
        
        var hList = [h0]
        for proj in altupProjections {
            hList.append(proj(h0))
        }
        
        var hStacked = stacked(hList, axis: 0)
        let mags = sqrt(hStacked[1...].square().mean(axis: -1, keepDims: true))
        let maxMags = maximum(mags, MLXArray(Float.leastNormalMagnitude))
        hStacked[1...] = hStacked[1...] * (targetMagnitude / maxMags)
        
        for (i, layer) in layers.enumerated() {
            let perLayerInput = actualPerLayerInputs?[0..., 0..., i, 0...]
            
            let isGlobal = config.layerTypes[i] == "full_attention"
            
            let localMask: MLXArray?
            if let mask = mask {
                localMask = mask
            } else if isGlobal {
                localMask = fullMask
            } else {
                localMask = slidingWindowMask
            }
            
            hStacked = layer(
                hStacked,
                mask: localMask,
                cache: cache[layerIdxToCacheIdx[i]],
                perLayerInput: perLayerInput
            )
        }
        
        // Per-layer inputs to single output
        let targetMagnitude2 = sqrt(hStacked[0].square().mean(axis: -1, keepDims: true))
        for (i, proj) in altupUnembedProjections.enumerated() {
            hStacked[i + 1] = proj(hStacked[i + 1])
        }
        
        let mags2 = sqrt(hStacked[1...].square().mean(axis: -1, keepDims: true))
        let maxMags2 = maximum(mags2, MLXArray(Float.leastNormalMagnitude))
        hStacked[1...] = hStacked[1...] * (targetMagnitude2 / maxMags2)
        
        let hMean = hStacked.mean(axis: 0)
        return norm(hMean)
    }
}

// MARK: - Language Model

public class Gemma3nLanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo(key: "model") public var model: Gemma3Model
    
    public let config: Gemma3nTextConfiguration
    public var kvHeads: [Int]
    let finalLogitSoftcapping: Float
    
    public init(_ config: Gemma3nTextConfiguration) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self.kvHeads = Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        self._model.wrappedValue = Gemma3Model(config)
        super.init()
    }
    
    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        var caches: [any KVCache] = []
        for layerType in config.layerTypes[..<model.firstKVSharedLayerIdx] {
            if layerType == "full_attention" {
                caches.append(StandardKVCache())
            } else if layerType == "sliding_attention" {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            } else {
                fatalError("Unknown layer type: \(layerType)")
            }
        }
        return caches
    }
    
    func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> LMOutput {
        let out = model(
            inputs,
            inputsEmbeds: inputsEmbeds,
            mask: mask,
            cache: cache,
            perLayerInputs: perLayerInputs
        )
        
        var finalLogits = model.embedTokens.asLinear(out)
        
        if finalLogitSoftcapping > 0 {
            let scale = MLXArray(finalLogitSoftcapping)
            finalLogits = tanh(finalLogits / scale) * scale
        }
        
        return LMOutput(logits: finalLogits)
    }
    
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]
        
        for (k, v) in weights {
            if !k.starts(with: "language_model") {
                sanitizedWeights[k] = v
                continue
            }
            print("Key: \(k), Shape: \(v.shape)")
        }
        
        return sanitizedWeights
    }
}

//TODO: use helper functions from KVCache implementation
// Helper function to create attention mask
private func createAttentionMask(h: MLXArray, cache: [KVCache]) -> MLXArray? {
    // Implementation should create appropriate causal mask
    // This is a simplified version - actual implementation may need more sophistication
    let seqLength = h.shape[1]
    return createCausalMask(seqLength)
}

private func createCausalMask(_ seqLength: Int) -> MLXArray {
    let mask = tril(ones([seqLength, seqLength])) 
    return MLX.where(equal(mask, 0), MLXArray(Float.infinity * -1), MLXArray(0))
}