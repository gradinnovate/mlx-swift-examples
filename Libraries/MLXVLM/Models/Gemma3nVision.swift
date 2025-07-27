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

/// Nearest neighbor interpolation for upsampling feature maps
/// Optimized implementation using MLX Swift APIs
private func nearestInterpolate(_ input: MLXArray, targetSize: [Int]) -> MLXArray {
    let (batchSize, channels, currentH, currentW) = (
        input.shape[0], input.shape[1], input.shape[2], input.shape[3]
    )
    let (targetH, targetW) = (targetSize[0], targetSize[1])
    
    // If already the target size, return as-is
    if currentH == targetH && currentW == targetW {
        return input
    }
    
    
    // For exact integer scaling, use repeat operations
    if targetH % currentH == 0 && targetW % currentW == 0 {
        let scaleH = targetH / currentH
        let scaleW = targetW / currentW
        
        // Use repeated() free function from MLX Swift API
        var result = input
        if scaleH > 1 {
            result = repeated(result, count: scaleH, axis: 2)
        }
        if scaleW > 1 {
            result = repeated(result, count: scaleW, axis: 3)
        }
        
        return result
    }
    
    // For non-integer scaling, use coordinate mapping
    let scaleH = Float(currentH) / Float(targetH)
    let scaleW = Float(currentW) / Float(targetW)
    
    // Create coordinate arrays for target positions
    let yCoords = MLXArray(0..<targetH).asType(.float32) * scaleH
    let xCoords = MLXArray(0..<targetW).asType(.float32) * scaleW
    
    // Convert to integer indices (nearest neighbor)
    let yIndices = clip(yCoords.asType(.int32), min: 0, max: currentH - 1)
    let xIndices = clip(xCoords.asType(.int32), min: 0, max: currentW - 1)
    
    // Use take() function with proper indexing
    var outputs: [MLXArray] = []
    
    for b in 0..<batchSize {
        for c in 0..<channels {
            let inputSlice = input[b, c]  // Shape: [currentH, currentW]
            
            // Create meshgrid for coordinates
            let yGrid = broadcast(yIndices.expandedDimensions(axis: 1), to: [targetH, targetW])
            let xGrid = broadcast(xIndices.expandedDimensions(axis: 0), to: [targetH, targetW])
            
            // Convert 2D coordinates to flat indices
            let flatIndices = yGrid * currentW + xGrid
            
            // Flatten input and use take() to gather values
            let inputFlat = inputSlice.flattened()
            let sampledFlat = take(inputFlat, flatIndices.flattened())
            
            // Reshape back to target size
            let sampledSlice = sampledFlat.reshaped([targetH, targetW])
            outputs.append(sampledSlice)
        }
    }
    
    // Stack all slices and reshape to final format
    let result = stacked(outputs, axis: 0).reshaped([batchSize, channels, targetH, targetW])
    
    return result
}

private func numGroups(groupSize: Int?, channels: Int) -> Int {
    if groupSize == nil || groupSize == 0 {
        return 1  // Normal conv with 1 group
    }
    else if let groupSize = groupSize, groupSize == 1 {
        return channels // Depthwise conv
    }
    else {
        return channels
    }
    
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
    let applyAct: Bool
    
    @ParameterInfo(key: "weight") var weight: MLXArray
    
    public init(numChannels: Int, eps: Float = 1e-6, applyAct: Bool = true) {
        self.normalizedShape = [numChannels]
        self.eps = eps
        self.applyAct = applyAct
        
        self._weight.wrappedValue = ones([numChannels])
        super.init()
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
        var result = rmsNorm2d(xNCHW, normalizedShape: normalizedShape, weight: weight, eps: eps)
        
        // Apply dropout (Identity in Python, so no-op)
        // result = drop(result) // Identity in Python
        
        // Apply activation if specified
        if applyAct {
            result = gelu(result)
        }
        
        // Convert back to NHWC
        return result.transposed(0, 2, 3, 1)
    }
}

// MARK: - Layer Scale 2D

private class LayerScale2d: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray
    let inplace: Bool
    
    init(dim: Int, initValues: Float = 1e-5, inplace: Bool = false) {
        self.inplace = inplace
        self._gamma.wrappedValue = MLXArray(initValues) * ones([dim])
        super.init()
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
    let padH = getSamePadding(inputSize: ih, kernelSize: kernelSize[0], stride: stride[0], dilation: dilation[0])
    let padW = getSamePadding(inputSize: iw, kernelSize: kernelSize[1], stride: stride[1], dilation: dilation[1])
    
    // MLX pad format: [(low, high), (low, high), ...] for each axis
    let padWidths: [IntOrPair] = [
        IntOrPair((0, 0)),  // No padding for batch dimension
        IntOrPair((padH / 2, padH - padH / 2)),  // Height padding
        IntOrPair((padW / 2, padW - padW / 2)),  // Width padding
        IntOrPair((0, 0)),  // No padding for channel dimension
    ]
    
    return padded(x, widths: padWidths, mode: .constant, value: MLXArray(value))
}

// MARK: - Conv2d Same

private class Conv2dSame: Conv2d {
    let kernelSizeArray: [Int]
    let strideArray: [Int]
    let dilationArray: [Int]
    
    override init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
        dilation: IntOrPair = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.kernelSizeArray = [kernelSize.first, kernelSize.second]
        self.strideArray = [stride.first, stride.second]
        self.dilationArray = [dilation.first, dilation.second]
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
        
    }
    
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = padSame(x, kernelSize: kernelSizeArray, stride: strideArray, dilation: dilationArray)
        return super.callAsFunction(padded)
    }
}

// MARK: - ConvNormAct

private class ConvNormAct: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "bn") var bn: Gemma3nRMSNorm2d
    
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
                padding: IntOrPair(padding),
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
                padding: IntOrPair(padding),
                dilation: IntOrPair(dilation),
                groups: groups,
                bias: bias
            )
        }
        
        self._bn.wrappedValue = Gemma3nRMSNorm2d(numChannels: outChs, eps: eps, applyAct: applyAct)
        super.init()

        
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let c = conv(x)
        return bn(c)
    }
}

// MARK: - Universal Inverted Residual

private class UniversalInvertedResidual: Module, UnaryLayer {
    @ModuleInfo(key: "dw_start") var dwStart: ConvNormAct?
    @ModuleInfo(key: "pw_exp") var pwExp: ConvNormAct
    @ModuleInfo(key: "dw_mid") var dwMid: ConvNormAct?
    @ModuleInfo(key: "pw_proj") var pwProj: ConvNormAct
    @ModuleInfo(key: "layer_scale") var layerScale: LayerScale2d?
    
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
        self._pwExp.wrappedValue = ConvNormAct(
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
        
        self._pwProj.wrappedValue = ConvNormAct(
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
        super.init()
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

private class EdgeResidual: Module, UnaryLayer {
    @ModuleInfo(key: "conv_exp") var convExp: Conv2dSame
    @ModuleInfo(key: "bn1") var bn1: Gemma3nRMSNorm2d
    @ModuleInfo(key: "conv_pwl") var convPwl: Conv2d
    @ModuleInfo(key: "bn2") var bn2: Gemma3nRMSNorm2d
    
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
        
        self._convExp.wrappedValue = Conv2dSame(
            inputChannels: inChs,
            outputChannels: midChs,
            kernelSize: IntOrPair(expKernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(0),
            dilation: IntOrPair(dilation),
            groups: groups,
            bias: false
        )
        
        self._bn1.wrappedValue = Gemma3nRMSNorm2d(numChannels: midChs, eps: 1e-05)
        
        let paddingPwl = (pwKernelSize - 1) / 2
        self._convPwl.wrappedValue = Conv2d(
            inputChannels: midChs,
            outputChannels: outChs,
            kernelSize: IntOrPair(pwKernelSize),
            padding: IntOrPair(paddingPwl),
            bias: false
        )
        
        self._bn2.wrappedValue = Gemma3nRMSNorm2d(numChannels: outChs, eps: 1e-05, applyAct: false)
        super.init()
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

private class QuerySequence: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d

    //use create_conv2d interface in vision.py
    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        depthwise: Bool = false,
        bias: Bool = false
    ) {
        if depthwise {
            let padding = Int((kernelSize - 1)/2) * dilation
            self._proj.wrappedValue = Conv2d(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                dilation: IntOrPair(dilation),
                groups: inputChannels,
                bias: bias
            )
        } else {
            let padding = Int((kernelSize - 1)/2) * dilation
            self._proj.wrappedValue = Conv2d(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                dilation: IntOrPair(dilation),
                bias: bias
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return proj(x)
    }
}

private class KeySequence: Module {
    @ModuleInfo(key: "down_conv") var downConv: Conv2d?
    @ModuleInfo(key: "norm") var norm: Gemma3nRMSNorm2d?
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(
        inputChannels: Int,
        numHeads: Int=8,
        keyDim: Int=64,
        valueDim: Int = 64,
        kvStride: Int = 1,
        dilation: Int = 1,
        dwKernelSize: Int = 3,
        numChannels: Int = 0,
        eps: Float = 1e-6,
        applyAct: Bool = false
    ) {

        if kvStride > 1 {
            let padding = Int((dwKernelSize - 1)/2) * dilation
            self._downConv.wrappedValue = Conv2d(
                inputChannels: inputChannels,
                outputChannels: inputChannels,
                kernelSize: IntOrPair(dwKernelSize),
                stride: IntOrPair(kvStride),
                padding: IntOrPair(padding),
                dilation: IntOrPair(dilation),  
                groups: inputChannels,
                bias: false
            )
            self._norm.wrappedValue = Gemma3nRMSNorm2d(numChannels: numChannels, eps: eps, applyAct: applyAct)

        } else {
            self._downConv.wrappedValue = nil
            self._norm.wrappedValue = nil
        }
        
        self._proj.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: keyDim,
            kernelSize: IntOrPair(1),
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        
        if let downConv = downConv {
            result = downConv(result)
        }
        if let norm = norm {
            result = norm(result)
        }
        result = proj(result)
        
        return result
    }
}

private class ValueSequence: Module {
    @ModuleInfo(key: "down_conv") var downConv: Conv2d?
    @ModuleInfo(key: "norm") var norm: Gemma3nRMSNorm2d?
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(
        inputChannels: Int,
        numHeads: Int = 8,
        keyDim: Int = 64,
        valueDim: Int = 64,
        kvStride: Int = 1,
        dilation: Int = 1,
        dwKernelSize: Int = 3,
        numChannels: Int = 0,
        eps: Float = 1e-6,
        applyAct: Bool = false
    ) {
        if kvStride > 1 {
            let padding = Int((dwKernelSize - 1)/2) * dilation
            self._downConv.wrappedValue = Conv2d(
                inputChannels: inputChannels,
                outputChannels: inputChannels,
                kernelSize: IntOrPair(dwKernelSize),
                stride: IntOrPair(kvStride),
                padding: IntOrPair(padding),
                dilation: IntOrPair(dilation),
                groups: inputChannels,
                bias: false
            )
            self._norm.wrappedValue = Gemma3nRMSNorm2d(numChannels: numChannels, eps: eps, applyAct: applyAct)

        } else {
            self._downConv.wrappedValue = nil
        }
        
        self._proj.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: valueDim,
            kernelSize: IntOrPair(1),
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        
        if let downConv = downConv {
            result = downConv(result)
        }
        if let norm = norm {
            result = norm(result)
        }
        result = proj(result)
        
        return result
    }
}

private class OutputSequence: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    let projDrop: Dropout

    init(
        inputChannels: Int,
        outputChannels: Int,
        projDrop: Float = 0.0
    ) {
        self._proj.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair(1),
            stride: IntOrPair(1),
            bias: false
        )
        
        self.projDrop = Dropout(p: projDrop)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = proj(x)
        result = projDrop(result)
        return result
    }
}


private class MultiQueryAttention2d: Module {
    @ModuleInfo(key: "query") var query: QuerySequence
    @ModuleInfo(key: "key") var key: KeySequence
    @ModuleInfo(key: "value") var value: ValueSequence
    @ModuleInfo(key: "output") var output: OutputSequence
    
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
        self._query.wrappedValue = QuerySequence(
            inputChannels: dim,
            outputChannels: numHeads * keyDim,
            kernelSize: 1
        )
        
        // Key projection
        self._key.wrappedValue = KeySequence(
            inputChannels: dim,
            numHeads: numHeads,
            keyDim: keyDim,
            valueDim: valueDim,
            kvStride: kvStride,
            dilation: dilation,
            dwKernelSize: dwKernelSize,
            numChannels: dim,
            eps: 1e-6,
            applyAct: false
        )
        
        // Value projection
        self._value.wrappedValue = ValueSequence(
            inputChannels: dim,
            numHeads: numHeads,
            keyDim: keyDim,
            valueDim: valueDim,
            kvStride: kvStride,
            dilation: dilation,
            dwKernelSize: dwKernelSize,
            numChannels: dim,
            eps: 1e-6,
            applyAct: false
        )
        
        // Output projection
        self._output.wrappedValue = OutputSequence(
            inputChannels: valueDim * numHeads,
            outputChannels: dimOut,
            projDrop: projDrop
        )
        super.init()
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
        
        let k = key(x)
        let kReshaped = reshapeInput(k)
        
        let v = value(x)
        let vReshaped = reshapeInput(v)
        
        let o: MLXArray
        if fusedAttn {
            o = MLXFast.scaledDotProductAttention(
                queries: qReshaped,
                keys: kReshaped,
                values: vReshaped,
                scale: 1.0 / sqrt(Float(qReshaped.shape.last!)),
                mask: .none
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

private class MobileAttention: Module, UnaryLayer {
    @ModuleInfo(key: "norm") var norm: Gemma3nRMSNorm2d
    @ModuleInfo(key: "attn") var attn: MultiQueryAttention2d
    @ModuleInfo(key: "layer_scale") var layerScale: LayerScale2d?
    
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
        self._norm.wrappedValue = Gemma3nRMSNorm2d(
            numChannels: inChs,
            eps: 1e-05,
            applyAct: false
        )
        
        // Attention layer
        if useMultiQuery {
            self._attn.wrappedValue = MultiQueryAttention2d(
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
        super.init()
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
    @ModuleInfo(key: "ffn") var ffn: UniversalInvertedResidual
    @ModuleInfo(key: "norm") var norm: Gemma3nRMSNorm2d
    @ModuleInfo(key: "avg_pool") var avgPool: AvgPool2d?
    
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
        
        self._ffn.wrappedValue = UniversalInvertedResidual(
            inChs: self.inChannels,
            outChs: self.outChannels,
            dwKernelSizeStart: 0,
            dwKernelSizeMid: 0,
            noskip: self.noskip,
            expRatio: expansionRatio,
            normLayer: Gemma3nRMSNorm2d.self,
            layerScaleInitValue: useLayerScale ? layerScaleInitValue : nil
        )
        
        self._norm.wrappedValue = Gemma3nRMSNorm2d(numChannels: outChannels, eps: 1e-6, applyAct: false)
        
        // Note: Pooling logic would be handled in MediaProcessing during preprocessing
        // following the critical guideline
        super.init()
    }
    
    func callAsFunction(_ inputs: [MLXArray]) -> MLXArray {
        // Convert from NHWC to NCHW for processing
        let inputsNCHW = inputs.map { $0.transposed(0, 3, 1, 2) }
        
        // Get highest resolution (assuming first input is highest resolution)
        let highResolution = Array(inputsNCHW[0].shape[2...])
        
        // Resize all inputs to the highest resolution if needed
        var resizedInputs: [MLXArray] = []
        for img in inputsNCHW {
            let currentResolution = Array(img.shape[2...])
            if currentResolution[0] < highResolution[0] || currentResolution[1] < highResolution[1] {
                let resized = nearestInterpolate(img, targetSize: highResolution)
                resizedInputs.append(resized)
            } else {
                resizedInputs.append(img)
            }
        }
        
        // Concatenate along channel dimension
        let channelCatImgs = concatenated(resizedInputs, axis: 1)
        
        // Apply FFN (convert back to NHWC for processing)
        let channelCatImgsNHWC = channelCatImgs.transposed(0, 2, 3, 1)
        var img = ffn(channelCatImgsNHWC)
        
        // Apply normalization if needed
        if noskip {
            img = norm(img)
        }
        
        // Convert back to NCHW for final resolution adjustment
        img = img.transposed(0, 3, 1, 2)
        
        // Final output resolution adjustment (matching Python implementation)
        let currentResolution = Array(img.shape[2...])
        if currentResolution[0] != outputResolution.0 || currentResolution[1] != outputResolution.1 {
            if currentResolution[0] % outputResolution.0 != 0 || currentResolution[1] % outputResolution.1 != 0 {
                // Use nearest interpolation for now (can be upgraded to bicubic later)
                img = nearestInterpolate(img, targetSize: [outputResolution.0, outputResolution.1])
            } else {
                // Use average pooling for divisible ratios
                let hStrides = currentResolution[0] / outputResolution.0
                let wStrides = currentResolution[1] / outputResolution.1
                
                // Convert to NHWC for pooling operations in MLX Swift
                let imgNHWC = img.transposed(0, 2, 3, 1)
                
                // Use MLX Swift's built-in average pooling
                let poolingLayer = AvgPool2d(kernelSize: [hStrides, wStrides], stride: [hStrides, wStrides])
                let pooled = poolingLayer(imgNHWC)
                
                img = pooled.transposed(0, 3, 1, 2)
            }
        }
        
        // Convert back to NHWC format for output
        return img.transposed(0, 2, 3, 1)
    }
}

// MARK: - Vision Tower

private class VisionTower: Module {
    @ModuleInfo(key: "conv_stem") var convStem: ConvNormAct
    @ModuleInfo(key: "blocks") var blocks: [[Module]]
    @ModuleInfo(key: "msfa") var msfa: MobileNetV5MultiScaleFusionAdapter
    
    let numFeatures: Int
    let headHiddenSize: Int
    let msfaIndices: (Int, Int)
    let msfaOutputResolution: (Int, Int)
    
    init(config: Gemma3nVisionConfiguration) {
        self._convStem.wrappedValue = ConvNormAct(
            convCls: Conv2dSame.self,
            inChs: 3,
            outChs: 64,
            kernelSize: 3,
            stride: 2,
            padding: 0,
            bias: true,
            eps: 1e-05
        )
        
        self.msfaIndices = (3, 4)
        self.msfaOutputResolution = (16, 16)
        
        // Build blocks
        let (numFeatures, blocks) = Self.buildBlocks()
        self.numFeatures = numFeatures
        self.headHiddenSize = numFeatures
        self._blocks.wrappedValue = blocks
        
        self._msfa.wrappedValue = MobileNetV5MultiScaleFusionAdapter(
            inChs: [1920],
            outChs: 2048,
            outputResolution: msfaOutputResolution.0
        )
        super.init()
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
                if let unaryBlock = block as? any UnaryLayer {
                    x = unaryBlock(x)
                } else {
                    fatalError("Block must implement UnaryLayer")
                }
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
    @ModuleInfo(key: "timm_model") private var timmModel: VisionTower
    
    let modelType: String
    
    public init(config: Gemma3nVisionConfiguration) {
        self.modelType = config.modelType
        
        if !["gemma3", "gemma3_vision", "gemma3n_vision"].contains(modelType) {
            fatalError("Unsupported vision model type: \(modelType)")
        }
        
        self._timmModel.wrappedValue = VisionTower(config: config)
        super.init()
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
            if !k.starts(with: "vision_tower") {
                sanitizedWeights[k] = v
                continue
            }
            print("Key: \(k), Shape: \(v.shape)")
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
    // Stage 1: Edge Residuals
    let stage1: [BlockConfig] = [
        .edgeResidual(kernelSize: 3, filters: 128, strides: 2, expandRatio: 4.0, isMultiscale: false),
        .edgeResidual(kernelSize: 3, filters: 128, strides: 1, expandRatio: 4.0, isMultiscale: false),
        .edgeResidual(kernelSize: 3, filters: 128, strides: 1, expandRatio: 4.0, isMultiscale: false)
    ]
    
    // Stage 2: Universal Inverted Residuals
    var stage2: [BlockConfig] = [
        .universalInvertedResidual(startDw: 3, midDw: 5, filters: 256, strides: 2, expandRatio: 6.0, isMultiscale: false)
    ]
    stage2 += [
        .universalInvertedResidual(startDw: 5, midDw: 0, filters: 256, strides: 1, expandRatio: 4.0, isMultiscale: false),
        .universalInvertedResidual(startDw: 3, midDw: 0, filters: 256, strides: 1, expandRatio: 4.0, isMultiscale: false),
        .universalInvertedResidual(startDw: 5, midDw: 0, filters: 256, strides: 1, expandRatio: 4.0, isMultiscale: false),
        .universalInvertedResidual(startDw: 3, midDw: 0, filters: 256, strides: 1, expandRatio: 4.0, isMultiscale: false)
    ]
    
    // Stage 3: Universal Inverted Residuals with Multi-Query Attention
    var stage3: [BlockConfig] = [
        .universalInvertedResidual(startDw: 5, midDw: 5, filters: 640, strides: 2, expandRatio: 6.0, isMultiscale: false)
    ]
    stage3 += Array(repeating: .universalInvertedResidual(startDw: 5, midDw: 0, filters: 640, strides: 1, expandRatio: 4.0, isMultiscale: false), count: 7)
    stage3 += [
        .universalInvertedResidual(startDw: 0, midDw: 0, filters: 640, strides: 1, expandRatio: 1.0, isMultiscale: false)
    ]
    
    // Add 13 pairs of multiQueryAttention + universalInvertedResidual
    for _ in 0..<13 {
        stage3 += [
            .multiQueryAttention(numHeads: 12, kvDim: 64, kvStrides: 2, avgPoolKv: false, isMultiscale: false),
            .universalInvertedResidual(startDw: 0, midDw: 0, filters: 640, strides: 1, expandRatio: 2.0, isMultiscale: false)
        ]
    }
    stage3 += [
        .multiQueryAttention(numHeads: 12, kvDim: 64, kvStrides: 2, avgPoolKv: false, isMultiscale: false),
        .universalInvertedResidual(startDw: 0, midDw: 0, filters: 640, strides: 1, expandRatio: 2.0, isMultiscale: true)
    ]
    
    // Stage 4: Universal Inverted Residuals with Multi-Query Attention
    var stage4: [BlockConfig] = [
        .universalInvertedResidual(startDw: 5, midDw: 5, filters: 1280, strides: 2, expandRatio: 6.0, isMultiscale: false)
    ]
    
    // Add 18 pairs of multiQueryAttention + universalInvertedResidual
    for _ in 0..<18 {
        stage4 += [
            .multiQueryAttention(numHeads: 16, kvDim: 96, kvStrides: 1, avgPoolKv: false, isMultiscale: false),
            .universalInvertedResidual(startDw: 0, midDw: 0, filters: 1280, strides: 1, expandRatio: 2.0, isMultiscale: false)
        ]
    }
    stage4 += [
        .multiQueryAttention(numHeads: 16, kvDim: 96, kvStrides: 1, avgPoolKv: false, isMultiscale: false),
        .universalInvertedResidual(startDw: 0, midDw: 0, filters: 1280, strides: 1, expandRatio: 2.0, isMultiscale: true)
    ]
    
    return [stage1, stage2, stage3, stage4]
}
