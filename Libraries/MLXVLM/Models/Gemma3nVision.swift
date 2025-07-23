// Copyright © 2024 Apple Inc.

import CoreImage
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// Based on https://github.com/ml-explore/mlx-vlm/tree/main/mlx_vlm/models/gemma3n/vision.py
// Note: Following critical guideline - all interpolation is handled in MediaProcessing during preprocessing

// MARK: - Helper Functions

private func to2Tuple<T>(_ x: T) -> (T, T) {
    return (x, x)
}

private func numGroups(groupSize: Int?, channels: Int) -> Int {
    guard let groupSize = groupSize, groupSize > 0 else {
        return 1  // Normal conv with 1 group
    }
    // NOTE: groupSize == 1 -> depthwise conv
    assert(channels % groupSize == 0)
    return channels / groupSize
}

private func makeDivisible(_ v: Int, divisor: Int = 8, minValue: Int? = nil, roundLimit: Float = 0.9) -> Int {
    let minValue = minValue ?? divisor
    let newV = max(minValue, (v + divisor / 2) / divisor * divisor)
    // Make sure that round down does not go down by more than 10%
    if Float(newV) < roundLimit * Float(v) {
        return newV + divisor
    }
    return newV
}

// MARK: - RMS Normalization

public class Gemma3nRMSNorm2d: Module {
    let normalizedShape: [Int]
    let eps: Float
    
    @ParameterInfo var weight: MLXArray?
    
    public init(numChannels: Int, eps: Float = 1e-6, applyAct: Bool = true) {
        self.normalizedShape = [numChannels]
        self.eps = eps
        
        self.weight = ones([numChannels])
    }
    
    private func rmsNorm2d(
        _ x: MLXArray,
        normalizedShape: [Int],
        weight: MLXArray?,
        eps: Float
    ) -> MLXArray {
        assert(normalizedShape.count == 1)
        let dtype = x.dtype
        let v = pow(x, 2).mean(axis: 1, keepDims: true)
        var result = x * rsqrt(v + eps)
        
        if let weight = weight {
            let weightReshaped = weight.reshaped([1, -1, 1, 1])
            result = result.asType(dtype) * weightReshaped
        }
        
        return result
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Convert from NHWC to NCHW for processing
        let xNCHW = x.transposed(0, 3, 1, 2)
        let result = rmsNorm2d(xNCHW, normalizedShape: normalizedShape, weight: weight, eps: eps)
        // Convert back to NHWC
        return result.transposed(0, 2, 3, 1)
    }
}

// MARK: - Layer Scale 2D

private class LayerScale2d: Module {
    @ParameterInfo var gamma: MLXArray
    let inplace: Bool
    
    init(dim: Int, initValues: Float = 1e-5, inplace: Bool = false) {
        self.inplace = inplace
        self.gamma = MLXArray(initValues) * ones([dim])
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x * gamma
    }
}

// MARK: - Convolution Helpers

private func getSamePadding(
    inputSize: Int,
    kernelSize: Int,
    stride: Int,
    dilation: Int = 1
) -> Int {
    let effectiveKernelSize = dilation * (kernelSize - 1) + 1
    let outputSize = (inputSize + stride - 1) / stride
    let totalPadding = max(0, (outputSize - 1) * stride + effectiveKernelSize - inputSize)
    return totalPadding
}

private func padSame(
    _ x: MLXArray,
    kernelSize: [Int],
    stride: [Int],
    dilation: [Int] = [1, 1],
    value: Float = 0
) -> MLXArray {
    let ih = x.shape[1]
    let iw = x.shape[2]
    let padH = getSamePadding(ih, kernelSize: kernelSize[0], stride: stride[0], dilation: dilation[0])
    let padW = getSamePadding(iw, kernelSize: kernelSize[1], stride: stride[1], dilation: dilation[1])
    
    // MLX pad format: [(low, high), (low, high), ...] for each axis
    let padWidths = [
        (0, 0),  // No padding for batch dimension
        (padH / 2, padH - padH / 2),  // Height padding
        (padW / 2, padW - padW / 2),  // Width padding
        (0, 0),  // No padding for channel dimension
    ]
    
    return padded(x, widths: padWidths, mode: .constant, value: value)
}

// MARK: - Conv2d Same

private class Conv2dSame: Conv2d {
    let kernelSize: [Int]
    
    override init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: IntOrPair,
        stride: IntOrPair = IntOrPair(1),
        padding: PaddingOrInt = PaddingOrInt(0),
        dilation: IntOrPair = IntOrPair(1),
        groups: Int = 1,
        bias: Bool = true
    ) {
        super.init(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
        
        self.kernelSize = [kernelSize.first, kernelSize.second]
    }
    
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = padSame(x, kernelSize: kernelSize, stride: [stride.first, stride.second], dilation: [dilation.first, dilation.second])
        return super.callAsFunction(padded)
    }
}

// MARK: - ConvNormAct

private class ConvNormAct: Module {
    @ModuleInfo var conv: Module
    @ModuleInfo var bn: Gemma3nRMSNorm2d
    
    init(
        convCls: Any.Type,
        inChs: Int,
        outChs: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = false,
        applyAct: Bool = true,
        eps: Float = 1e-6
    ) {
        if convCls is Conv2dSame.Type {
            self._conv.wrappedValue = Conv2dSame(
                inputChannels: inChs,
                outputChannels: outChs,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: PaddingOrInt(padding),
                dilation: IntOrPair(dilation),
                groups: groups,
                bias: bias
            )
        } else {
            self._conv.wrappedValue = Conv2d(
                inputChannels: inChs,
                outputChannels: outChs,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: PaddingOrInt(padding),
                dilation: IntOrPair(dilation),
                groups: groups,
                bias: bias
            )
        }
        
        self.bn = Gemma3nRMSNorm2d(numChannels: outChs, eps: eps, applyAct: applyAct)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = conv(x)
        return bn(c)
    }
}

// MARK: - Universal Inverted Residual

private class UniversalInvertedResidual: Module {
    @ModuleInfo var dwStart: Module?
    @ModuleInfo var pwExp: ConvNormAct
    @ModuleInfo var dwMid: Module?
    @ModuleInfo var pwProj: ConvNormAct
    @ModuleInfo var layerScale: Module?
    
    let hasSkip: Bool
    
    init(
        inChs: Int,
        outChs: Int,
        dwKernelSizeStart: Int = 0,
        dwKernelSizeMid: Int = 3,
        dwKernelSizeEnd: Int = 0,
        stride: Int = 1,
        dilation: Int = 1,
        groupSize: Int = 1,
        padType: String = "",
        noskip: Bool = false,
        expRatio: Float = 1.0,
        normLayer: Any.Type = Gemma3nRMSNorm2d.self,
        dropPathRate: Float = 0.0,
        layerScaleInitValue: Float? = 1e-5
    ) {
        self.hasSkip = (inChs == outChs && stride == 1) && !noskip
        
        if stride > 1 {
            assert(dwKernelSizeStart > 0 || dwKernelSizeMid > 0 || dwKernelSizeEnd > 0)
        }
        
        if dwKernelSizeStart > 0 {
            let dwStartStride = (dwKernelSizeMid == 0) ? stride : 1
            let dwStartGroups = numGroups(groupSize: groupSize, channels: inChs)
            self._dwStart.wrappedValue = ConvNormAct(
                convCls: Conv2d.self,
                inChs: inChs,
                outChs: inChs,
                kernelSize: dwKernelSizeStart,
                stride: dwStartStride,
                padding: (dwKernelSizeStart - 1) / 2,
                dilation: dilation,
                groups: dwStartGroups,
                bias: false,
                applyAct: false,
                eps: 1e-05
            )
        } else {
            self._dwStart.wrappedValue = nil
        }
        
        let midChs = makeDivisible(Int(Float(inChs) * expRatio))
        self.pwExp = ConvNormAct(
            convCls: Conv2d.self,
            inChs: inChs,
            outChs: midChs,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            groups: 1,
            bias: false,
            eps: 1e-05
        )
        
        if dwKernelSizeMid > 0 {
            let dwMidGroups = numGroups(groupSize: groupSize, channels: midChs)
            self._dwMid.wrappedValue = ConvNormAct(
                convCls: Conv2dSame.self,
                inChs: midChs,
                outChs: midChs,
                kernelSize: dwKernelSizeMid,
                stride: stride,
                padding: 0,
                dilation: dilation,
                groups: dwMidGroups,
                bias: false,
                eps: 1e-05
            )
        } else {
            self._dwMid.wrappedValue = nil
        }
        
        self.pwProj = ConvNormAct(
            convCls: Conv2d.self,
            inChs: midChs,
            outChs: outChs,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            groups: 1,
            bias: false,
            applyAct: false,
            eps: 1e-05
        )
        
        if let layerScaleInitValue = layerScaleInitValue {
            self._layerScale.wrappedValue = LayerScale2d(dim: outChs, initValues: layerScaleInitValue)
        } else {
            self._layerScale.wrappedValue = nil
        }
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = x
        var result = x
        
        if let dwStart = dwStart {
            result = dwStart(result)
        }
        
        result = pwExp(result)
        
        if let dwMid = dwMid {
            result = dwMid(result)
        }
        
        result = pwProj(result)
        
        if let layerScale = layerScale {
            result = layerScale(result)
        }
        
        if hasSkip {
            result = result + shortcut
        }
        
        return result
    }
}

// MARK: - Edge Residual

private class EdgeResidual: Module {
    @ModuleInfo var convExp: Conv2dSame
    @ModuleInfo var bn1: Gemma3nRMSNorm2d
    @ModuleInfo var convPwl: Conv2d
    @ModuleInfo var bn2: Gemma3nRMSNorm2d
    
    let hasSkip: Bool
    
    init(
        inChs: Int,
        outChs: Int,
        expKernelSize: Int = 3,
        stride: Int = 1,
        dilation: Int = 1,
        groupSize: Int = 0,
        padType: String = "",
        forceInChs: Int = 0,
        noskip: Bool = false,
        expandRatio: Float = 1.0,
        pwKernelSize: Int = 1,
        normLayer: Any.Type = Gemma3nRMSNorm2d.self
    ) {
        let midChs: Int
        if forceInChs > 0 {
            midChs = makeDivisible(Int(Float(forceInChs) * expandRatio))
        } else {
            midChs = makeDivisible(Int(Float(inChs) * expandRatio))
        }
        
        let groups = numGroups(groupSize: groupSize, channels: midChs)
        
        self.hasSkip = (inChs == outChs && stride == 1) && !noskip
        
        self.convExp = Conv2dSame(
            inputChannels: inChs,
            outputChannels: midChs,
            kernelSize: IntOrPair(expKernelSize),
            stride: IntOrPair(stride),
            padding: PaddingOrInt(0),
            dilation: IntOrPair(dilation),
            groups: groups,
            bias: false
        )
        
        self.bn1 = Gemma3nRMSNorm2d(numChannels: midChs, eps: 1e-05)
        
        let paddingPwl = (pwKernelSize - 1) / 2
        self.convPwl = Conv2d(
            inputChannels: midChs,
            outputChannels: outChs,
            kernelSize: IntOrPair(pwKernelSize),
            padding: PaddingOrInt(paddingPwl),
            bias: false
        )
        
        self.bn2 = Gemma3nRMSNorm2d(numChannels: outChs, eps: 1e-05, applyAct: false)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = x
        var result = convExp(x)
        result = bn1(result)
        result = convPwl(result)
        result = bn2(result)
        
        if hasSkip {
            result = result + shortcut
        }
        
        return result
    }
}

// MARK: - Multi-Query Attention 2D

private class MultiQueryAttention2d: Module {
    @ModuleInfo var query: Conv2d
    @ModuleInfo var key: Conv2d
    @ModuleInfo var keyNorm: Gemma3nRMSNorm2d?
    @ModuleInfo var value: Conv2d
    @ModuleInfo var valueNorm: Gemma3nRMSNorm2d?
    @ModuleInfo var output: Conv2d
    
    let numHeads: Int
    let queryStrides: (Int, Int)
    let kvStride: Int
    let fusedAttn: Bool
    let keyDim: Int
    let valueDim: Int
    let scale: Float
    
    init(
        dim: Int,
        dimOut: Int? = nil,
        numHeads: Int = 8,
        keyDim: Int = 64,
        valueDim: Int = 64,
        queryStrides: (Int, Int) = (1, 1),
        kvStride: Int = 1,
        dilation: Int = 1,
        padding: String = "",
        dwKernelSize: Int = 3,
        attnDrop: Float = 0.0,
        projDrop: Float = 0.0
    ) {
        let dimOut = dimOut ?? dim
        self.numHeads = numHeads
        self.queryStrides = queryStrides
        self.kvStride = kvStride
        self.fusedAttn = true
        self.keyDim = keyDim
        self.valueDim = valueDim
        let headDim = keyDim
        self.scale = pow(Float(headDim), -0.5)
        
        // Query projection
        self.query = Conv2d(
            inputChannels: dim,
            outputChannels: numHeads * keyDim,
            kernelSize: IntOrPair(1)
        )
        
        // Key projection
        if kvStride > 1 {
            // For downsampling, we would need depthwise convolution + norm
            // Simplified version for now
            self.key = Conv2d(
                inputChannels: dim,
                outputChannels: keyDim,
                kernelSize: IntOrPair(1),
                bias: false
            )
            self.keyNorm = Gemma3nRMSNorm2d(numChannels: dim, eps: 1e-6, applyAct: false)
        } else {
            self.key = Conv2d(
                inputChannels: dim,
                outputChannels: keyDim,
                kernelSize: IntOrPair(1),
                bias: false
            )
            self.keyNorm = nil
        }
        
        // Value projection
        if kvStride > 1 {
            self.value = Conv2d(
                inputChannels: dim,
                outputChannels: valueDim,
                kernelSize: IntOrPair(1),
                bias: false
            )
            self.valueNorm = Gemma3nRMSNorm2d(numChannels: dim, eps: 1e-6, applyAct: false)
        } else {
            self.value = Conv2d(
                inputChannels: dim,
                outputChannels: valueDim,
                kernelSize: IntOrPair(1),
                bias: false
            )
            self.valueNorm = nil
        }
        
        // Output projection
        self.output = Conv2d(
            inputChannels: valueDim * numHeads,
            outputChannels: dimOut,
            kernelSize: IntOrPair(1),
            stride: IntOrPair(1),
            bias: false
        )
    }
    
    private func reshapeInput(_ t: MLXArray) -> MLXArray {
        let s = t.shape
        return t.reshaped([s[0], -1, s[3]]).expandedDimensions(axis: 1)
    }
    
    private func reshapeProjectedQuery(_ t: MLXArray, numHeads: Int, keyDim: Int) -> MLXArray {
        let (B, H, W, C) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3])
        return t.reshaped([B, H * W, numHeads, keyDim]).transposed(0, 2, 1, 3)
    }
    
    private func reshapeOutput(_ t: MLXArray, numHeads: Int, hPx: Int, wPx: Int) -> MLXArray {
        let (B, NH, L, D) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3])
        return t.transposed(0, 2, 1, 3).reshaped([B, hPx, wPx, NH * D])
    }
    
    func callAsFunction(_ x: MLXArray, attnMask: MLXArray? = nil) -> MLXArray {
        let (B, H, W, C) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        
        let q = query(x)
        let qReshaped = reshapeProjectedQuery(q, numHeads: numHeads, keyDim: keyDim)
        
        var k = key(x)
        if let keyNorm = keyNorm {
            k = keyNorm(k)
        }
        let kReshaped = reshapeInput(k)
        
        var v = value(x)
        if let valueNorm = valueNorm {
            v = valueNorm(v)
        }
        let vReshaped = reshapeInput(v)
        
        let o: MLXArray
        if fusedAttn {
            o = MLXFast.scaledDotProductAttention(
                queries: qReshaped,
                keys: kReshaped,
                values: vReshaped,
                scale: 1.0 / sqrt(Float(qReshaped.shape.last!))
            )
        } else {
            fatalError("Unfused attention not implemented")
        }
        
        let oReshaped = reshapeOutput(
            o, 
            numHeads: numHeads, 
            hPx: H / queryStrides.0, 
            wPx: W / queryStrides.1
        )
        
        return output(oReshaped)
    }
}

// MARK: - Mobile Attention

private class MobileAttention: Module {
    @ModuleInfo var norm: Gemma3nRMSNorm2d
    @ModuleInfo var attn: MultiQueryAttention2d
    @ModuleInfo var layerScale: Module?
    
    let hasSkip: Bool
    let queryStrides: (Int, Int)
    
    init(
        inChs: Int,
        outChs: Int,
        stride: Int = 1,
        dwKernelSize: Int = 3,
        dilation: Int = 1,
        groupSize: Int = 1,
        padType: String = "",
        numHeads: Int = 8,
        keyDim: Int = 64,
        valueDim: Int = 64,
        useMultiQuery: Bool = true,
        queryStrides: (Int, Int) = (1, 1),
        kvStride: Int = 1,
        cpeDwKernelSize: Int = 3,
        noskip: Bool = false,
        actLayer: Any? = nil,
        aaLayer: Any? = nil,
        dropPathRate: Float = 0.0,
        attnDrop: Float = 0.0,
        projDrop: Float = 0.0,
        layerScaleInitValue: Float? = 1e-5,
        useBias: Bool = false
    ) {
        self.hasSkip = (stride == 1 && inChs == outChs) && !noskip
        self.queryStrides = queryStrides
        
        // Normalization layer
        self.norm = Gemma3nRMSNorm2d(
            numChannels: inChs,
            eps: 1e-05,
            applyAct: false
        )
        
        // Attention layer
        if useMultiQuery {
            self.attn = MultiQueryAttention2d(
                dim: inChs,
                dimOut: outChs,
                numHeads: numHeads,
                keyDim: keyDim,
                valueDim: valueDim,
                queryStrides: queryStrides,
                kvStride: kvStride,
                dilation: dilation,
                padding: padType,
                dwKernelSize: dwKernelSize,
                attnDrop: attnDrop,
                projDrop: projDrop
            )
        } else {
            fatalError("Non-multi-query attention not implemented")
        }
        
        // Layer scaling
        if let layerScaleInitValue = layerScaleInitValue {
            self._layerScale.wrappedValue = LayerScale2d(dim: outChs, initValues: layerScaleInitValue)
        } else {
            self._layerScale.wrappedValue = nil
        }
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = x
        var result = norm(x)
        result = attn(result)
        
        if let layerScale = layerScale {
            result = layerScale(result)
        }
        
        if hasSkip {
            result = result + shortcut
        }
        
        return result
    }
}

// MARK: - MobileNet V5 Multi-Scale Fusion Adapter

private class MobileNetV5MultiScaleFusionAdapter: Module {
    @ModuleInfo var ffn: UniversalInvertedResidual
    @ModuleInfo var norm: Gemma3nRMSNorm2d
    @ModuleInfo var avgPool: AvgPool2d?
    
    let inChannels: Int
    let outChannels: Int
    let outputResolution: (Int, Int)
    let expansionRatio: Float
    let interpolationMode: String
    let useLayerScale: Bool
    let layerScaleInitValue: Float
    let noskip: Bool
    
    init(
        inChs: [Int],
        outChs: Int,
        outputResolution: Int,
        expansionRatio: Float = 2.0,
        interpolationMode: String = "nearest",
        useLayerScale: Bool = false,
        layerScaleInitValue: Float = 1e-5,
        noskip: Bool = true
    ) {
        self.inChannels = inChs.reduce(0, +)
        self.outChannels = outChs
        self.outputResolution = (outputResolution, outputResolution)
        self.expansionRatio = expansionRatio
        self.interpolationMode = interpolationMode
        self.useLayerScale = useLayerScale
        self.layerScaleInitValue = layerScaleInitValue
        self.noskip = noskip
        
        self.ffn = UniversalInvertedResidual(
            inChs: self.inChannels,
            outChs: self.outChannels,
            dwKernelSizeStart: 0,
            dwKernelSizeMid: 0,
            expRatio: expansionRatio,
            normLayer: Gemma3nRMSNorm2d.self,
            noskip: self.noskip,
            layerScaleInitValue: useLayerScale ? layerScaleInitValue : nil
        )
        
        self.norm = Gemma3nRMSNorm2d(numChannels: outChannels, eps: 1e-6, applyAct: false)
        
        // Note: Pooling logic would be handled in MediaProcessing during preprocessing
        // following the critical guideline
    }
    
    func callAsFunction(_ inputs: [MLXArray]) -> MLXArray {
        // Convert from NHWC to NCHW for processing
        let inputsNCHW = inputs.map { $0.transposed(0, 3, 1, 2) }
        
        // Get highest resolution
        let highResolution = inputsNCHW[0].shape[2...]
        
        // For MLX Swift implementation, we assume all preprocessing including
        // resizing and interpolation has been done in MediaProcessing
        // This follows the critical guideline to avoid interpolation in the neural network
        
        // Concatenate along channel dimension
        let channelCatImgs = concatenated(inputsNCHW, axis: 1)
        
        // Apply FFN (convert back to NHWC for processing)
        let channelCatImgsNHWC = channelCatImgs.transposed(0, 2, 3, 1)
        var img = ffn(channelCatImgsNHWC)
        
        // Apply normalization if needed
        if noskip {
            img = norm(img)
        }
        
        return img
    }
}

// MARK: - Vision Tower

private class VisionTower: Module {
    @ModuleInfo var convStem: ConvNormAct
    @ModuleInfo var blocks: [[Module]]
    @ModuleInfo var msfa: MobileNetV5MultiScaleFusionAdapter
    
    let numFeatures: Int
    let headHiddenSize: Int
    let msfaIndices: (Int, Int)
    let msfaOutputResolution: (Int, Int)
    
    init(config: Gemma3nVisionConfiguration) {
        self.convStem = ConvNormAct(
            convCls: Conv2dSame.self,
            inChs: 3,
            outChs: 64,
            kernelSize: 3,
            stride: 2,
            padding: 0,
            eps: 1e-05,
            bias: true
        )
        
        self.msfaIndices = (3, 4)
        self.msfaOutputResolution = (16, 16)
        
        // Build blocks
        let (numFeatures, blocks) = Self.buildBlocks()
        self.numFeatures = numFeatures
        self.headHiddenSize = numFeatures
        self._blocks.wrappedValue = blocks
        
        self.msfa = MobileNetV5MultiScaleFusionAdapter(
            inChs: [1920],
            outChs: 2048,
            outputResolution: msfaOutputResolution.0
        )
    }
    
    private static func buildBlocks() -> (Int, [[Module]]) {
        let def = gemma3nMobilenetDef()
        var blocks: [[Module]] = []
        var inChs = 64  // From conv stem
        
        for (stage, blockConfigs) in def.enumerated() {
            var blockGroup: [Module] = []
            
            for config in blockConfigs {
                switch config {
                case .edgeResidual(let kernelSize, let filters, let strides, let expandRatio, _):
                    let block = EdgeResidual(
                        inChs: inChs,
                        outChs: filters,
                        expKernelSize: kernelSize,
                        stride: strides,
                        expandRatio: expandRatio
                    )
                    inChs = filters
                    blockGroup.append(block)
                    
                case .universalInvertedResidual(let startDw, let midDw, let filters, let strides, let expandRatio, _):
                    let block = UniversalInvertedResidual(
                        inChs: inChs,
                        outChs: filters,
                        dwKernelSizeStart: startDw,
                        dwKernelSizeMid: midDw,
                        stride: strides,
                        expRatio: expandRatio
                    )
                    inChs = filters
                    blockGroup.append(block)
                    
                case .multiQueryAttention(let numHeads, let kvDim, let kvStrides, _, _):
                    let block = MobileAttention(
                        inChs: inChs,
                        outChs: inChs,
                        stride: 1,
                        numHeads: numHeads,
                        keyDim: kvDim,
                        valueDim: kvDim,
                        kvStride: kvStrides
                    )
                    blockGroup.append(block)
                }
            }
            
            blocks.append(blockGroup)
        }
        
        return (inChs, blocks)
    }
    
    func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool? = nil) -> MLXArray {
        var featIdx = 0
        // Convert from NCHW to NHWC
        var x = x.transposed(0, 2, 3, 1)
        x = convStem(x)
        var intermediates: [MLXArray] = []
        
        if msfaIndices.0 == featIdx || msfaIndices.1 == featIdx {
            intermediates.append(x)
        }
        
        // Process through blocks
        for blockGroup in blocks {
            featIdx += 1
            for block in blockGroup {
                x = block(x)
            }
            
            if msfaIndices.0 == featIdx || msfaIndices.1 == featIdx {
                intermediates.append(x)
            }
        }
        
        x = msfa(intermediates)
        return x
    }
}

// MARK: - Vision Model

public class Gemma3nVisionModel: Module {
    @ModuleInfo var timmModel: VisionTower
    
    let modelType: String
    
    public init(config: Gemma3nVisionConfiguration) {
        self.modelType = config.modelType
        
        if !["gemma3", "gemma3_vision", "gemma3n_vision"].contains(modelType) {
            fatalError("Unsupported model type: \(modelType)")
        }
        
        self.timmModel = VisionTower(config: config)
    }
    
    public func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool? = nil) -> MLXArray {
        return timmModel(x, outputHiddenStates: outputHiddenStates)
    }
    
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights: [String: MLXArray] = [:]
        var skipTranspose = false
        
        // Check if weights are already in correct format
        if let sampleWeight = weights["vision_tower.timm_model.blocks.0.0.conv_exp.weight"] {
            let (_, H, _, C) = (sampleWeight.shape[0], sampleWeight.shape[1], sampleWeight.shape[2], sampleWeight.shape[3])
            if C > H {
                skipTranspose = true
            }
        }
        
        for (k, v) in weights {
            // PyTorch conv2d weight: [out_channels, in_channels, kH, kW]
            // MLX conv2d weight: [out_channels, kH, kW, in_channels]
            if (k.contains("conv") && k.contains("weight")) || (k.contains("attn") && k.contains("proj.weight")) {
                if v.shape.count == 4 && !skipTranspose {
                    sanitizedWeights[k] = v.transposed(0, 2, 3, 1)
                } else {
                    sanitizedWeights[k] = v
                }
            } else {
                sanitizedWeights[k] = v
            }
        }
        
        return sanitizedWeights
    }
}

// MARK: - Block Configuration Types

private enum BlockConfig {
    case edgeResidual(kernelSize: Int, filters: Int, strides: Int, expandRatio: Float, isMultiscale: Bool)
    case universalInvertedResidual(startDw: Int, midDw: Int, filters: Int, strides: Int, expandRatio: Float, isMultiscale: Bool)
    case multiQueryAttention(numHeads: Int, kvDim: Int, kvStrides: Int, avgPoolKv: Bool, isMultiscale: Bool)
}

private func gemma3nMobilenetDef() -> [[BlockConfig]] {
    return [
        // Stage 1: Edge Residuals
        [
            .edgeResidual(kernelSize: 3, filters: 128, strides: 2, expandRatio: 4.0, isMultiscale: false),
            .edgeResidual(kernelSize: 3, filters: 128, strides: 1, expandRatio: 4.0, isMultiscale: false),
            .edgeResidual(kernelSize: 3, filters: 128, strides: 1, expandRatio: 4.0, isMultiscale: false)
        ],
        // Stage 2: Universal Inverted Residuals
        [
            .universalInvertedResidual(startDw: 3, midDw: 5, filters: 256, strides: 2, expandRatio: 6.0, isMultiscale: false)
        ] + [
            .universalInvertedResidual(startDw: 5, midDw: 0, filters: 256, strides: 1, expandRatio: 4.0, isMultiscale: false),
            .universalInvertedResidual(startDw: 3, midDw: 0, filters: 256, strides: 1, expandRatio: 4.0, isMultiscale: false),
            .universalInvertedResidual(startDw: 5, midDw: 0, filters: 256, strides: 1, expandRatio: 4.0, isMultiscale: false),
            .universalInvertedResidual(startDw: 3, midDw: 0, filters: 256, strides: 1, expandRatio: 4.0, isMultiscale: false)
        ],
        // Stage 3: Universal Inverted Residuals with Multi-Query Attention
        [
            .universalInvertedResidual(startDw: 5, midDw: 5, filters: 640, strides: 2, expandRatio: 6.0, isMultiscale: false)
        ] + Array(repeating: .universalInvertedResidual(startDw: 5, midDw: 0, filters: 640, strides: 1, expandRatio: 4.0, isMultiscale: false), count: 7) + [
            .universalInvertedResidual(startDw: 0, midDw: 0, filters: 640, strides: 1, expandRatio: 1.0, isMultiscale: false)
        ] + Array(0..<13).flatMap { _ in [
            .multiQueryAttention(numHeads: 12, kvDim: 64, kvStrides: 2, avgPoolKv: false, isMultiscale: false),
            .universalInvertedResidual(startDw: 0, midDw: 0, filters: 640, strides: 1, expandRatio: 2.0, isMultiscale: false)
        ]} + [
            .multiQueryAttention(numHeads: 12, kvDim: 64, kvStrides: 2, avgPoolKv: false, isMultiscale: false),
            .universalInvertedResidual(startDw: 0, midDw: 0, filters: 640, strides: 1, expandRatio: 2.0, isMultiscale: true)
        ],
        // Stage 4: Universal Inverted Residuals with Multi-Query Attention
        [
            .universalInvertedResidual(startDw: 5, midDw: 5, filters: 1280, strides: 2, expandRatio: 6.0, isMultiscale: false)
        ] + Array(0..<18).flatMap { _ in [
            .multiQueryAttention(numHeads: 16, kvDim: 96, kvStrides: 1, avgPoolKv: false, isMultiscale: false),
            .universalInvertedResidual(startDw: 0, midDw: 0, filters: 1280, strides: 1, expandRatio: 2.0, isMultiscale: false)
        ]} + [
            .multiQueryAttention(numHeads: 16, kvDim: 96, kvStrides: 1, avgPoolKv: false, isMultiscale: false),
            .universalInvertedResidual(startDw: 0, midDw: 0, filters: 1280, strides: 1, expandRatio: 2.0, isMultiscale: true)
        ]
    ]
}