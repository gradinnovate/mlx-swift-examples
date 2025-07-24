// Copyright © 2024 Apple Inc.

import CoreImage
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Foundation

// Based on https://github.com/ml-explore/mlx-vlm/tree/main/mlx_vlm/models/gemma3n/audio.py

// MARK: - Helper Functions

private func convertTorchToMLXPadWidth(_ padding: [Int], _ inputShape: [Int]) -> [IntOrPair] {
    let ndim = inputShape.count
    var padWidth = Array(repeating: IntOrPair((0, 0)), count: ndim)
    
    if ndim >= 1 && padding.count >= 2 {
        padWidth[ndim - 1] = IntOrPair((padding[0], padding[1]))
    }
    if ndim >= 2 && padding.count >= 4 {
        padWidth[ndim - 2] = IntOrPair((padding[2], padding[3]))
    }
    if ndim >= 3 && padding.count >= 6 {
        padWidth[ndim - 3] = IntOrPair((padding[4], padding[5]))
    }
    if ndim >= 4 && padding.count >= 8 {
        padWidth[ndim - 4] = IntOrPair((padding[6], padding[7]))
    }
    
    return padWidth
}

// MARK: - Audio Relative Position Embedding

private class Gemma3nAudioRelativePositionEmbedding: Module {
    @ModuleInfo(key: "pos_proj") var posProj: Linear
    let invTimescales: MLXArray
    
    let config: Gemma3nAudioConfiguration
    let numHeads: Int
    let channels: Int
    let headDim: Int
    let maxBackward: Int
    let maxForward: Int
    
    init(config: Gemma3nAudioConfiguration) {
        self.config = config
        self.numHeads = config.confNumAttentionHeads
        self.channels = config.hiddenSize
        self.headDim = channels / numHeads
        
        self.maxBackward = max(0, config.confAttentionContextLeft - 1)
        self.maxForward = config.confAttentionContextRight
        
        self._posProj.wrappedValue = Linear(
            channels, numHeads * headDim, bias: false
        )
        
        let minTimescale: Float = 1.0
        let maxTimescale: Float = 1.0e4
        let numTimescales = channels / 2
        let logTimescaleIncrement = log(maxTimescale / minTimescale) / max(Float(numTimescales - 1), 1)
        
        let invTimescales = minTimescale * exp(
            MLXArray(0..<numTimescales).asType(.float32) * (-logTimescaleIncrement)
        )
        
        self.invTimescales = invTimescales.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    }
    
    private func getTimingSignal1dPos(_ position: MLXArray, dtype: DType) -> MLXArray {
        assert(position.ndim == 2)
        let position = position.expandedDimensions(axis: -1).asType(.float32)
        
        let scaledTime = position * invTimescales
        let timingSignal = concatenated([sin(scaledTime), cos(scaledTime)], axis: -1)
        return timingSignal.asType(dtype)
    }
    
    private func relativeShift(
        _ termBdBeforeShift: MLXArray,
        batchSize: Int,
        numHeads: Int,
        numQueryBlocks: Int,
        queryBlockSize: Int,
        keyContextSize: Int,
        maxSpanPlus1: Int
    ) -> MLXArray {
        let padAmountLastDim = (keyContextSize + 1) - maxSpanPlus1
        
        let paddingTuple = [0, padAmountLastDim]
        let padWidths = convertTorchToMLXPadWidth(paddingTuple, termBdBeforeShift.shape)
        
        let termBdPadded = padded(termBdBeforeShift, widths: padWidths, mode: .constant)
        
        let termBdReshaped = termBdPadded.reshaped([
            batchSize,
            numHeads,
            numQueryBlocks,
            queryBlockSize * (keyContextSize + 1)
        ])
        
        let termBdSliced = termBdReshaped[0..., 0..., 0..., ..<(queryBlockSize * keyContextSize)]
        
        let termBdShifted = termBdSliced.reshaped([
            batchSize,
            numHeads,
            numQueryBlocks,
            queryBlockSize,
            keyContextSize
        ])
        
        return termBdShifted
    }
    
    func callAsFunction(_ queries: MLXArray, _ keys: MLXArray) -> MLXArray {
        let (batchSize, numQueryBlocks, queryBlockSize, numHeads, headDim) = (
            queries.shape[0], queries.shape[1], queries.shape[2], queries.shape[3], queries.shape[4]
        )
        let keyContextSize = keys.shape[2]
        
        let posIndices = MLXArray(maxBackward...(-maxForward-1))[.stride(by: -1)].expandedDimensions(axis: 0)
        let maxSpanPlus1 = posIndices.shape[1]
        
        let sinEmbTimingSignal = getTimingSignal1dPos(posIndices, dtype: queries.dtype)
        
        let projectedSinEmb = posProj(sinEmbTimingSignal)
        let sinEmb = projectedSinEmb.reshaped([1, maxSpanPlus1, numHeads, headDim]).squeezed(axis: 0)
        
        // Query-Key content interaction
        let queriesP = queries.transposed(0, 3, 1, 2, 4)
        let keysPT = keys.transposed(0, 3, 1, 4, 2)
        let termAc = matmul(queriesP, keysPT)
        
        // Query-Position interaction
        let qTransposed = queries.transposed(0, 3, 1, 2, 4)
        let sTransposed = sinEmb.transposed(1, 2, 0)
        
        let qReshaped = qTransposed.reshaped([
            batchSize, numHeads, numQueryBlocks * queryBlockSize, headDim
        ])
        
        let termBdUnshifedMatmul = matmul(qReshaped, sTransposed)
        
        let termBdUnshifed = termBdUnshifedMatmul.reshaped([
            batchSize, numHeads, numQueryBlocks, queryBlockSize, maxSpanPlus1
        ])
        
        let termBdShifted = relativeShift(
            termBdUnshifed,
            batchSize: batchSize,
            numHeads: numHeads,
            numQueryBlocks: numQueryBlocks,
            queryBlockSize: queryBlockSize,
            keyContextSize: keyContextSize,
            maxSpanPlus1: maxSpanPlus1
        )
        
        return termAc + termBdShifted
    }
}

// MARK: - Audio Attention

private class Gemma3nAudioAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear  
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "relative_position_embedding") var relativePositionEmbedding: Gemma3nAudioRelativePositionEmbedding
    @ParameterInfo(key: "per_dim_scale") var perDimScale: MLXArray
    let localCausalValidMask: MLXArray
    let softcap: MLXArray
    
    let config: Gemma3nAudioConfiguration
    let numHeads: Int
    let hiddenSize: Int
    let headDim: Int
    let chunkSize: Int
    let maxFutureHorizon: Int
    let maxPastHorizon: Int
    let attentionInvalidLogitsValue: Float
    let attentionLogitsSoftCap: Float
    let contextSize: Int
    let qScale: Float
    
    init(config: Gemma3nAudioConfiguration) {
        self.config = config
        self.numHeads = config.confNumAttentionHeads
        self.hiddenSize = config.hiddenSize
        self.headDim = hiddenSize / numHeads
        self.chunkSize = config.confAttentionChunkSize
        self.maxFutureHorizon = config.confAttentionContextRight
        self.maxPastHorizon = max(0, config.confAttentionContextLeft - 1)
        self.attentionInvalidLogitsValue = config.confAttentionInvalidLogitsValue
        self.attentionLogitsSoftCap = config.confAttentionLogitCap
        self.contextSize = chunkSize + maxPastHorizon + maxFutureHorizon
        
        self._relativePositionEmbedding.wrappedValue = Gemma3nAudioRelativePositionEmbedding(config: config)
        self._perDimScale.wrappedValue = zeros([headDim])
        
        self._qProj.wrappedValue = Linear(
            hiddenSize, numHeads * headDim, bias: false
        )
        self._kProj.wrappedValue = Linear(
            hiddenSize, numHeads * headDim, bias: false
        )
        self._vProj.wrappedValue = Linear(
            hiddenSize, numHeads * headDim, bias: false
        )
        
        let qScale = pow(Float(headDim), -0.5)
        let rSoftplus0 = Float(1.0 / log(2.0))
        self.qScale = qScale * rSoftplus0
        
        // Create causal masks
        let lowerCausalMask = tril(
            MLXArray.ones([contextSize, chunkSize], type: Bool.self), k: 0
        ).transposed()
        
        let upperCausalMask = tril(
            MLXArray.ones([chunkSize, contextSize], type: Bool.self),
            k: maxPastHorizon + maxFutureHorizon
        )
        
        let localCausalValidMask = MLXArray.ones([chunkSize, contextSize], type: Bool.self)
        self.localCausalValidMask = logicalAnd(
            logicalAnd(localCausalValidMask, lowerCausalMask),
            upperCausalMask
        )
        
        self.softcap = MLXArray(attentionLogitsSoftCap)
    }
    
    private func padDim1(
        _ x: MLXArray,
        dim10Val: Int,
        dim11Val: Int
    ) -> MLXArray {
        var paddingTuple = Array(repeating: 0, count: x.ndim * 2)
        let dimIdxFromEnd = x.ndim - 2
        let startIdxForDim = 2 * dimIdxFromEnd
        paddingTuple[startIdxForDim] = dim10Val
        paddingTuple[startIdxForDim + 1] = dim11Val
        
        let padWidths = convertTorchToMLXPadWidth(paddingTuple, x.shape)
        return padded(x, widths: padWidths, mode: .constant)
    }
    
    private func convertToBlock(
        _ x: MLXArray,
        paddingVal: Float = 0.0
    ) -> MLXArray {
        let shape = x.shape
        let (b, t) = (shape[0], shape[1])
        let numBlocks = (t + chunkSize - 1) / chunkSize
        
        let paddingLen = numBlocks * chunkSize - t
        var result = x
        if paddingLen > 0 {
            result = padDim1(result, dim10Val: 0, dim11Val: paddingLen)
        }
        
        let permuteShape = [b, numBlocks, chunkSize] + Array(shape[2...])
        return result.reshaped(permuteShape)
    }
    
    private func unfoldMLX(_ x: MLXArray, dimension: Int, size: Int, step: Int) -> MLXArray {
        let shape = x.shape
        let dimSize = shape[dimension]
        let numWindows = (dimSize - size) / step + 1
        
        var windows: [MLXArray] = []
        for i in 0..<numWindows {
            let startIdx = i * step
            let endIdx = startIdx + size
            
            var slices: [any MLXArrayIndex] = Array(repeating: MLXEllipsisIndex.ellipsis, count: shape.count)
            slices[dimension] = startIdx..<endIdx
            
            windows.append(x[slices])
        }
        
        return stacked(windows, axis: dimension + 1)
    }
    
    private func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let padLeft = maxPastHorizon
        let padRight = maxFutureHorizon + chunkSize - 1
        let padded = padDim1(x, dim10Val: padLeft, dim11Val: padRight)
        
        let frameLen = contextSize
        let frameStep = chunkSize
        
        var xUnfolded = unfoldMLX(padded, dimension: 1, size: frameLen, step: frameStep)
        
        if x.ndim > 2 && xUnfolded.ndim > 3 {
            xUnfolded = xUnfolded.transposed(0, 2, 1, 3, 4)
        }
        
        return xUnfolded
    }
    
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let queryStates = qProj(x).reshaped(x.shape + [numHeads, headDim])
        let keyStates = kProj(x).reshaped(x.shape + [numHeads, headDim])  
        let valueStates = vProj(x).reshaped(x.shape + [numHeads, headDim])
        
        let perDimScaleSp = logAddExp(perDimScale, MLXArray(0.0))
        let broadcastShape = [1, 1, 1, headDim]
        let perDimScaleBroadcast = perDimScaleSp.reshaped(broadcastShape)
        let scaledQueries = queryStates * qScale * perDimScaleBroadcast
        
        let (batchSize, qTime) = (scaledQueries.shape[0], scaledQueries.shape[1])
        
        let queryBlocks = convertToBlock(scaledQueries)
        let keyBlocks = extractBlockContext(keyStates)
        let valueBlocks = extractBlockContext(valueStates)
        let numQueryBlocks = queryBlocks.shape[1]
        
        // Create validity mask
        let originalValidMask = logicalNot(mask)
        var extractedValidMaskBlocks = extractBlockContext(originalValidMask).transposed(0, 2, 1)
        
        if extractedValidMaskBlocks.ndim == 4 &&
           extractedValidMaskBlocks.shape[0] == batchSize &&
           extractedValidMaskBlocks.shape[1] == numQueryBlocks &&
           extractedValidMaskBlocks.shape[2] * extractedValidMaskBlocks.shape[3] == contextSize {
            extractedValidMaskBlocks = extractedValidMaskBlocks.reshaped([
                batchSize, numQueryBlocks, contextSize
            ])
        }
        
        let conditionFromInputValidity = extractedValidMaskBlocks
            .expandedDimensions(axis: 1)
            .expandedDimensions(axis: -2)
        
        let conditionFromCausality = localCausalValidMask
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
        
        let finalConditionForWhere = logicalAnd(
            conditionFromInputValidity,
            conditionFromCausality
        )
        
        // Compute logits
        let logits = relativePositionEmbedding(queryBlocks, keyBlocks)
        
        // Apply softcapping
        var cappedLogits = logits / softcap
        cappedLogits = tanh(cappedLogits)
        cappedLogits = cappedLogits * softcap
        
        // Apply mask
        let maskedLogits = MLX.where(
            finalConditionForWhere,
            cappedLogits,
            MLXArray(attentionInvalidLogitsValue)
        )
        
        let probabilities = softmax(maskedLogits.asType(Float32.self), axis: -1)
            .asType(valueBlocks.dtype)
        
        // Compute attention output
        let (bDim, nDim, uDim, wDim, cDim) = (
            probabilities.shape[0], probabilities.shape[1], probabilities.shape[2],
            probabilities.shape[3], probabilities.shape[4]
        )
        let hDim = valueBlocks.shape.last!
        
        let probBun = probabilities.transposed(0, 2, 1, 3, 4).reshaped([-1, wDim, cDim])
        let vBun = valueBlocks.transposed(0, 1, 3, 2, 4).reshaped([-1, cDim, hDim])
        let resultBmm = matmul(probBun, vBun)
        
        var contextVectors = resultBmm.reshaped([bDim, uDim, nDim, wDim, hDim])
            .transposed(0, 1, 3, 2, 4)
        
        contextVectors = contextVectors.reshaped([
            batchSize, numQueryBlocks * chunkSize, numHeads, headDim
        ])
        
        return contextVectors[0..., ..<qTime, 0..., 0...]
    }
}

// MARK: - Cumulative Group Norm

private class Gemma3nCumulativeGroupNorm: Module {
    @ParameterInfo var weight: MLXArray?
    @ParameterInfo var bias: MLXArray?
    
    let numChannels: Int
    let featureDims: [Int]
    let eps: Float
    let useScale: Bool
    let useBias: Bool
    let reductionAxes: [Int]
    
    init(
        numChannels: Int,
        featureDims: [Int],
        eps: Float = 1e-3,
        useScale: Bool = true,
        useBias: Bool = false
    ) {
        self.numChannels = numChannels
        self.featureDims = featureDims
        self.eps = eps
        self.useScale = useScale
        self.useBias = useBias
        
        if useScale {
            self.weight = ones([numChannels])
        } else {
            self.weight = nil
        }
        
        if useBias {
            self.bias = zeros([numChannels])
        } else {
            self.bias = nil
        }
        
        self.reductionAxes = Array(2..<(2 + featureDims.count + 1))
    }
    
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let expectedInputSuffix = featureDims + [numChannels]
        let actualSuffix = Array(x.shape[2...])
        
        guard actualSuffix == expectedInputSuffix else {
            fatalError("Input shape suffix \(actualSuffix) doesn't match expected \(expectedInputSuffix)")
        }
        
        if let mask = mask {
            guard mask.shape == Array(x.shape[0..<2]) else {
                fatalError("Mask shape \(mask.shape) must match input batch/time dimensions \(Array(x.shape[0..<2]))")
            }
        }
        
        let inputDtype = x.dtype
        let calcDtype: DType = .float32
        let xCalc = x.asType(calcDtype)
        
        let maskCalc: MLXArray
        if let mask = mask {
            let maskSuffixShape = Array(repeating: 1, count: expectedInputSuffix.count)
            maskCalc = mask.reshaped(mask.shape + maskSuffixShape).asType(calcDtype)
        } else {
            maskCalc = ones(like: xCalc).asType(calcDtype)
        }
        
        let xMaskedForSum = xCalc * maskCalc
        
        // Cumulative statistics calculation
        let sumValuesAtT = xMaskedForSum.sum(axes: reductionAxes, keepDims: true)
        let cumSumValues = cumsum(sumValuesAtT, axis: 1)
        
        let elementsInGroupAtT = maskCalc.sum(axes: reductionAxes, keepDims: true)
        let cumCountElements = cumsum(elementsInGroupAtT, axis: 1)
        let safeCumCountElements = clip(cumCountElements, min: 1)
        
        let cumMean = cumSumValues / safeCumCountElements
        
        let squaredDiffFromMean = pow(xCalc - cumMean, 2)
        let sumSqDiffAtT = (squaredDiffFromMean * maskCalc).sum(axes: reductionAxes, keepDims: true)
        let cumSumSqDiff = cumsum(sumSqDiffAtT, axis: 1)
        
        let cumVariance = cumSumSqDiff / safeCumCountElements
        
        var normalizedX = (xCalc - cumMean) * rsqrt(cumVariance + eps)
        
        if useScale, let weight = weight {
            let scale = weight.asType(calcDtype)
            let scaleViewShape = Array(repeating: 1, count: x.ndim - 1) + [numChannels]
            normalizedX = normalizedX * scale.reshaped(scaleViewShape)
        }
        
        if useBias, let bias = bias {
            let biasValue = bias.asType(calcDtype)
            let biasViewShape = Array(repeating: 1, count: x.ndim - 1) + [numChannels]
            normalizedX = normalizedX + biasValue.reshaped(biasViewShape)
        }
        
        let finalOutput = normalizedX * maskCalc
        return finalOutput.asType(inputDtype)
    }
}

// MARK: - SSCP Conv Block

private class Gemma3nAudioSSCPConvBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: Gemma3nCumulativeGroupNorm
    
    let config: Gemma3nAudioConfiguration
    let manualPadding: [Int]
    
    init(
        idx: Int,
        inputFreqDim: Int,
        config: Gemma3nAudioConfiguration,
        manualPadding: [Int] = [0, 0, 0, 0]
    ) {
        self.config = config
        self.manualPadding = manualPadding
        
        let inChannels = idx == 0 ? 1 : config.sscpConvChannelSize[idx - 1]
        let outChannels = config.sscpConvChannelSize[idx]
        let (kernelH, kernelW) = (config.sscpConvKernelSize[idx][0], config.sscpConvKernelSize[idx][1])
        let (strideH, strideW) = (config.sscpConvStrideSize[idx][0], config.sscpConvStrideSize[idx][1])
        
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair((kernelH, kernelW)),
            stride: IntOrPair((strideH, strideW)),
            padding: IntOrPair(0),
            bias: false
        )
        
        let fInPadded = inputFreqDim + manualPadding[0] + manualPadding[1]
        let fOutConv = (fInPadded - kernelW) / strideW + 1
        
        self._norm.wrappedValue = Gemma3nCumulativeGroupNorm(
            numChannels: outChannels,
            featureDims: [fOutConv],
            eps: config.sscpConvEps,
            useScale: true,
            useBias: false
        )
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padWidths = convertTorchToMLXPadWidth(manualPadding, x.shape)
        let audioEncodingsPadded = padded(x, widths: padWidths, mode: .constant)
        
        let audioEncodingsConv = conv(audioEncodingsPadded.transposed(0, 2, 3, 1))
        let xNormed = norm(audioEncodingsConv)
        let audioEncodingsNormed = xNormed.transposed(0, 3, 1, 2)
        
        return relu(audioEncodingsNormed)
    }
}

// MARK: - Sub Sample Conv Projection

private class Gemma3nAudioSubSampleConvProjection: Module {
    @ModuleInfo(key: "conv0") var conv0: Gemma3nAudioSSCPConvBlock
    @ModuleInfo(key: "conv1") var conv1: Gemma3nAudioSSCPConvBlock
    @ModuleInfo(key: "input_proj_linear") var inputProjLinear: Linear
    
    let config: Gemma3nAudioConfiguration
    let inputProjInFeatures: Int
    
    init(config: Gemma3nAudioConfiguration) {
        self.config = config
        
        var currentFForBlockInput = config.inputFeatSize
        var calculatedBlockPadding: [[Int]] = []
        var calculatedFOutDims: [Int] = []
        
        for i in 0..<2 {
            let (kernelH, kernelW) = (config.sscpConvKernelSize[i][0], config.sscpConvKernelSize[i][1])
            let (strideH, strideW) = (config.sscpConvStrideSize[i][0], config.sscpConvStrideSize[i][1])
            
            let padTTop = 0
            let padTBottom = kernelH - 1
            let padFLeft = 1
            let padFRight = 1
            
            let manualPaddingTuple = [padFLeft, padFRight, padTTop, padTBottom]
            calculatedBlockPadding.append(manualPaddingTuple)
            
            let fInPadded = currentFForBlockInput + padFLeft + padFRight
            let fOutAfterConv = (fInPadded - kernelW) / strideW + 1
            calculatedFOutDims.append(fOutAfterConv)
            currentFForBlockInput = fOutAfterConv
        }
        
        self._conv0.wrappedValue = Gemma3nAudioSSCPConvBlock(
            idx: 0,
            inputFreqDim: config.inputFeatSize,
            config: config,
            manualPadding: calculatedBlockPadding[0]
        )
        self._conv1.wrappedValue = Gemma3nAudioSSCPConvBlock(
            idx: 1,
            inputFreqDim: calculatedFOutDims[0],
            config: config,
            manualPadding: calculatedBlockPadding[1]
        )
        
        let finalCOut = config.sscpConvChannelSize.last!
        let finalFOut = calculatedFOutDims.last!
        self.inputProjInFeatures = finalCOut * finalFOut
        
        self._inputProjLinear.wrappedValue = Linear(
            inputProjInFeatures, config.hiddenSize, bias: false
        )
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let audioEncodingsReshaped = x.expandedDimensions(axis: 1)
        var result = conv0(audioEncodingsReshaped)
        result = conv1(result)
        
        let (b, cOut, tOut, fOut) = (result.shape[0], result.shape[1], result.shape[2], result.shape[3])
        let xTransposed = result.transposed(0, 2, 3, 1)
        let outputFlattened = xTransposed.reshaped([b, tOut, fOut * cOut])
        
        return inputProjLinear(outputFlattened)
    }
}

// MARK: - Conformer Attention

private class Gemma3nAudioConformerAttention: Module {
    @ModuleInfo(key: "pre_attn_norm") var preAttnNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "attn") var attn: Gemma3nAudioAttention
    @ModuleInfo(key: "post") var post: Linear
    @ModuleInfo(key: "post_norm") var postNorm: Gemma3nRMSNorm
    
    let config: Gemma3nAudioConfiguration
    let gradientClipping: MLXArray
    
    init(config: Gemma3nAudioConfiguration) {
        self.config = config
        
        let headDim = config.hiddenSize / config.confNumAttentionHeads
        let postInFeatures = config.hiddenSize
        
        self.gradientClipping = MLXArray(config.gradientClipping)
        
        self._preAttnNorm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize)
        self._attn.wrappedValue = Gemma3nAudioAttention(config: config)
        self._post.wrappedValue = Linear(postInFeatures, config.hiddenSize, bias: false)
        self._postNorm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize)
    }
    
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let audioEncodingsInputToAttn = x
        let clippedX = clip(x, min: -gradientClipping, max: gradientClipping)
        let audioEncodingsNorm = preAttnNorm(clippedX)
        let audioEncodingsAttnOut = attn(audioEncodingsNorm, mask: mask)
        
        let (b, t, numHeads, headDim) = (
            audioEncodingsAttnOut.shape[0], audioEncodingsAttnOut.shape[1],
            audioEncodingsAttnOut.shape[2], audioEncodingsAttnOut.shape[3]
        )
        let audioEncodingsReshaped = audioEncodingsAttnOut.reshaped([b, t, numHeads * headDim])
        
        let postResult = post(audioEncodingsReshaped)
        let clippedPost = clip(postResult, min: -gradientClipping, max: gradientClipping)
        
        return audioEncodingsInputToAttn + postNorm(clippedPost)
    }
}

// MARK: - Conformer Feed Forward

private class Gemma3nAudioConformerFeedForward: Module {
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "ffw_layer_1") var ffwLayer1: Linear
    @ModuleInfo(key: "ffw_layer_2") var ffwLayer2: Linear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: Gemma3nRMSNorm
    
    let config: Gemma3nAudioConfiguration
    let gradientClipping: MLXArray
    let postLayerScale: MLXArray
    
    init(config: Gemma3nAudioConfiguration) {
        self.config = config
        
        self.gradientClipping = MLXArray(config.gradientClipping)
        
        self._preLayerNorm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize)
        self._ffwLayer1.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 4, bias: false)
        self._ffwLayer2.wrappedValue = Linear(config.hiddenSize * 4, config.hiddenSize, bias: false)
        self._postLayerNorm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize)
        self.postLayerScale = MLXArray(config.confResidualWeight)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        let clippedX = clip(x, min: -gradientClipping, max: gradientClipping)
        var result = preLayerNorm(clippedX)
        result = ffwLayer1(result)
        result = silu(result)
        result = ffwLayer2(result)
        result = clip(result, min: -gradientClipping, max: gradientClipping)
        result = postLayerNorm(result)
        
        return residual + (result * postLayerScale)
    }
}

// MARK: - Conformer Light Conv1d

private class Gemma3nAudioConformerLightConv1d: Module {
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: Linear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: Conv1d
    @ModuleInfo(key: "conv_norm") var convNorm: Gemma3nRMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: Linear
    
    let config: Gemma3nAudioConfiguration
    let gradientClipping: MLXArray
    let causalPadding: Int
    
    init(config: Gemma3nAudioConfiguration) {
        self.config = config
        
        self._preLayerNorm.wrappedValue = Gemma3nRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps
        )
        self._linearStart.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize * 2, bias: false
        )
        self._depthwiseConv1d.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.confConvKernelSize,
            stride: 1,
            padding: 0,
            groups: config.hiddenSize,
            bias: false
        )
        self.gradientClipping = MLXArray(config.gradientClipping)
        self._convNorm.wrappedValue = Gemma3nRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps
        )
        self._linearEnd.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: false
        )
        
        self.causalPadding = config.confConvKernelSize - 1
    }
    
    func callAsFunction(_ audioEncodings: MLXArray) -> MLXArray {
        let audioEncodingsResidual = audioEncodings
        
        var result = preLayerNorm(audioEncodings)
        result = linearStart(result)
        result = glu(result, axis: -1)
        
        // Apply manual causal padding for Conv1d
        let paddingTuple = [causalPadding, 0]
        let padWidths = convertTorchToMLXPadWidth(paddingTuple, result.shape)
        let resultPadded = padded(result, widths: padWidths, mode: .constant)
        
        result = depthwiseConv1d(resultPadded)
        result = clip(result, min: -gradientClipping, max: gradientClipping)
        result = convNorm(result)
        result = silu(result)
        result = linearEnd(result)
        
        return result + audioEncodingsResidual
    }
}

// MARK: - Conformer Block

private class Gemma3nAudioConformerBlock: Module {
    @ModuleInfo(key: "ffw_layer_start") var ffwLayerStart: Gemma3nAudioConformerFeedForward
    @ModuleInfo(key: "attention") var attention: Gemma3nAudioConformerAttention
    @ModuleInfo(key: "lconv1d") var lconv1d: Gemma3nAudioConformerLightConv1d
    @ModuleInfo(key: "ffw_layer_end") var ffwLayerEnd: Gemma3nAudioConformerFeedForward
    @ModuleInfo(key: "norm") var norm: Gemma3nRMSNorm
    
    let config: Gemma3nAudioConfiguration
    let gradientClipping: MLXArray
    
    init(config: Gemma3nAudioConfiguration) {
        self.config = config
        
        self._ffwLayerStart.wrappedValue = Gemma3nAudioConformerFeedForward(config: config)
        self._attention.wrappedValue = Gemma3nAudioConformerAttention(config: config)
        self._lconv1d.wrappedValue = Gemma3nAudioConformerLightConv1d(config: config)
        self._ffwLayerEnd.wrappedValue = Gemma3nAudioConformerFeedForward(config: config)
        self.gradientClipping = MLXArray(config.gradientClipping)
        self._norm.wrappedValue = Gemma3nRMSNorm(dimensions: config.hiddenSize)
    }
    
    func callAsFunction(_ audioEncodings: MLXArray, _ audioMelMask: MLXArray) -> MLXArray {
        var result = ffwLayerStart(audioEncodings)
        result = attention(result, mask: audioMelMask)
        
        let validityMaskForLconv = logicalNot(audioMelMask)
        let audioEncodingsForLconvInput = result * 
            validityMaskForLconv.expandedDimensions(axis: -1).asType(result.dtype)
        
        result = lconv1d(audioEncodingsForLconvInput)
        result = ffwLayerEnd(result)
        result = clip(result, min: -gradientClipping, max: gradientClipping)
        
        return norm(result)
    }
}

// MARK: - Audio Model

public class Gemma3nAudioModel: Module {
    @ModuleInfo(key: "subsample_conv_projection") private var subsampleConvProjection: Gemma3nAudioSubSampleConvProjection
    @ModuleInfo(key: "conformer") private var conformer: [Gemma3nAudioConformerBlock]
    
    let config: Gemma3nAudioConfiguration
    
    public init(config: Gemma3nAudioConfiguration) {
        self.config = config
        
        self._subsampleConvProjection.wrappedValue = Gemma3nAudioSubSampleConvProjection(config: config)
        self._conformer.wrappedValue = (0..<config.confNumHiddenLayers).map { _ in
            Gemma3nAudioConformerBlock(config: config)
        }
    }
    
    public func callAsFunction(
        _ audioMel: MLXArray,
        _ audioMelMask: MLXArray
    ) -> (MLXArray, MLXArray) {
        var audioEncodings = subsampleConvProjection(audioMel)
        let tSub = audioEncodings.shape[1]
        
        var timeStrideProduct = 1
        for stridePair in config.sscpConvStrideSize {
            timeStrideProduct *= stridePair[0]
        }
        
        // Create indices for gathering from the original mask
        var indices = MLXArray(0..<tSub) * timeStrideProduct
        indices = clip(indices, min: Int.min, max: audioMelMask.shape[1] - 1)
        
        if audioMelMask.ndim > 1 && indices.ndim == 1 {
            indices = indices.expandedDimensions(axis: 0)
            indices = broadcast(indices, to: [audioMelMask.shape[0], indices.shape[1]])
        }
        
        var currentMask = takeAlong(audioMelMask, indices, axis: 1)
        
        // Ensure mask length matches feature length
        if currentMask.shape[1] != tSub {
            if currentMask.shape[1] > tSub {
                currentMask = currentMask[0..., ..<tSub]
            } else {
                let paddingNeeded = tSub - currentMask.shape[1]
                let padWidths = convertTorchToMLXPadWidth([0, paddingNeeded], currentMask.shape)
                currentMask = padded(currentMask, widths: padWidths, mode: .constant)
            }
        }
        
        for block in conformer {
            audioEncodings = block(audioEncodings, currentMask)
        }
        
        if config.confReductionFactor > 1 {
            let step = config.confReductionFactor
            audioEncodings = audioEncodings[0..., .stride(from: 0, to: audioEncodings.shape[1], by: step), 0...]
            currentMask = currentMask[0..., .stride(from: 0, to: currentMask.shape[1], by: step)]
        }
        
        // Final masking adjustment
        if currentMask.shape[1] != audioEncodings.shape[1] {
            let targetLen = audioEncodings.shape[1]
            let maskCurrentLen = currentMask.shape[1]
            
            if targetLen > maskCurrentLen {
                let paddingNeeded = targetLen - maskCurrentLen
                let padWidths = convertTorchToMLXPadWidth([0, paddingNeeded], currentMask.shape)
                currentMask = padded(currentMask, widths: padWidths, mode: .constant)
            } else if maskCurrentLen > targetLen {
                currentMask = currentMask[0..., ..<targetLen]
            }
        }
        
        audioEncodings = MLX.where(currentMask.expandedDimensions(axis: -1), MLXArray(0.0), audioEncodings)
        return (audioEncodings, currentMask)
    }
    
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]
        
        for (k, v) in weights {
            if k.contains("conv.weight") {
                if checkArrayShape(v) {
                    sanitizedWeights[k] = v
                } else {
                    sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                }
            } else if k.contains("conv1d.weight") {
                if checkArrayShape(v) {
                    sanitizedWeights[k] = v
                } else {
                    sanitizedWeights[k] = v.transposed(0, 2, 1)
                }
            } else {
                sanitizedWeights[k] = v
            }
        }
        
        return sanitizedWeights
    }
}

// Helper function to check array shape
private func checkArrayShape(_ arr: MLXArray) -> Bool {
    let shape = arr.shape
    guard shape.count == 4 else { return false }
    
    let (outChannels, kH, kW, _) = (shape[0], shape[1], shape[2], shape[3])
    return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
}