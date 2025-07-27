# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the MLX Swift Examples repository - a collection of example applications and libraries demonstrating machine learning capabilities using MLX Swift. The project includes complete Swift packages for LLMs, VLMs, embedders, stable diffusion, and MNIST training.

## Build and Development Commands

### Testing
- `swift test` - Run all unit tests for the libraries
- Tests are located in `Tests/MLXLMTests/` and can be run via Xcode test plan at `Tests/mlx-libraries-Package.xctestplan`

### Building and Running Tools
- `./mlx-run <tool-name> [arguments]` - Primary way to build and run command-line tools
- `./mlx-run --debug <tool-name>` - Build and run in debug mode  
- `./mlx-run --release <tool-name>` - Build and run in release mode
- `./mlx-run --list` - List available build schemes

Available command-line tools:
- `llm-tool` - Generate text using various LLMs from Hugging Face
- `image-tool` - Generate images using stable diffusion models
- `mnist-tool` - Train LeNet on MNIST dataset
- `ExampleLLM` - Simplified API example for LLM interaction

#### Testing Gemma3n Vision-Language Model
For testing the newly implemented Gemma3n VLM model:
```bash
# Test with local model directory
./mlx-run llm-tool --model /path/to/gemma-3n-model --prompt "Describe this image" --image /path/to/image.jpg

# Test with Hugging Face model (if available)
./mlx-run llm-tool --model mlx-community/gemma-3n-4bit --prompt "Hello, how are you?"

# Test audio capabilities (if model supports audio)
./mlx-run llm-tool --model /path/to/gemma-3n-model --prompt "Transcribe this audio" --audio /path/to/audio.wav
```

Common debugging commands:
```bash
# Debug mode for more verbose output
./mlx-run --debug llm-tool --model /path/to/model --prompt "test"

# Check model configuration loading
./mlx-run llm-tool --model /path/to/model --list-config
```

### Code Formatting
- `swift-format format --in-place --recursive Libraries Tools Applications` - Format all code
- `pre-commit run --all-files` - Run all pre-commit hooks including formatting
- Install pre-commit: `pip install pre-commit && pre-commit install`
- May need: `brew install swift-format`

### Building Applications
Applications can be built and run from Xcode or using the `mlx-run` script. All iOS/macOS apps are in the `Applications/` directory.

## Architecture Overview

### Core Libraries Structure
The project is organized as multiple Swift packages:

- **MLXLMCommon** (`Libraries/MLXLMCommon/`) - Common API and infrastructure for both LLMs and VLMs
  - `ModelContainer` - Thread-safe model access wrapper using Swift actors
  - `Chat.swift` - Message types and chat formatting protocols 
  - `ModelFactory` - Abstract factory pattern for model loading
  - `UserInput` and processors for handling different input types
  - Tool calling support in `Tool/` directory

- **MLXLLM** (`Libraries/MLXLLM/`) - Large Language Model implementations
  - `LLMModelFactory` - Concrete factory for LLM loading with HuggingFace integration
  - `Models/` - 20+ LLM implementations (Llama, Qwen, Gemma, Phi, etc.)
  - Registry system mapping model types to implementations

- **MLXVLM** (`Libraries/MLXVLM/`) - Vision Language Model implementations  
  - `VLMModelFactory` - Factory for vision-language models
  - `Models/` - VLM implementations (Qwen2VL, PaliGemma, Idefics3, etc.)
  - `MediaProcessing.swift` - Image/video processing utilities

- **Other Libraries**:
  - **MLXEmbedders** - Text embedding models (BERT, NomicBert)
  - **StableDiffusion** - Image generation models
  - **MLXMNIST** - MNIST dataset handling and training

### Key Architectural Patterns

1. **Factory Pattern**: `ModelFactory` → `LLMModelFactory`/`VLMModelFactory` for model instantiation
2. **Registry Pattern**: Type registries map model configuration types to concrete implementations
3. **Actor-based Threading**: `ModelContainer` provides thread-safe model access
4. **Protocol-based Design**: `LanguageModel`, `UserInputProcessor`, `MessageGenerator` protocols
5. **Adapter Pattern**: LoRA adapters in `MLXLMCommon/Adapters/`

### Model Loading Flow
1. Download model from HuggingFace Hub
2. Decode `config.json` to determine model type
3. Registry lookup creates appropriate model instance
4. Load and apply weights to model
5. Create tokenizer and processor
6. Wrap in `ModelContainer` for thread safety

### Dependencies
- **MLX Swift** (v0.25.5+) - Core ML framework
- **swift-transformers** (v0.1.22+) - HuggingFace tokenizers
- **GzipSwift** - MNIST data decompression
- All libraries use `StrictConcurrency` Swift feature

## Development Guidelines

### MANDATORY API VERIFICATION WORKFLOW
1. **Step 1**: Before implementing any MLX Swift code, use context7 to query the specific API
2. **Step 2**: Verify parameter types, function signatures, and usage patterns from context7 results
3. **Step 3**: Only then proceed with implementation using the verified API information
4. **Step 4**: If compilation errors occur, return to context7 before making assumptions

### Model Implementation
- New model types go in respective `Models/` directories
- Register new models in factory registries (`LLMTypeRegistry.all()`)
- Follow existing configuration pattern with `Configuration.swift` structs
- Use proper weight loading in factory `_load()` methods

### Code Style
- Swift Concurrency (`async/await`) is used throughout
- Actor isolation for thread safety (`@Sendable` requirements)
- Protocol-oriented design preferred over inheritance
- Comprehensive error handling with custom error types

### Porting VLM Model Guidelines

#### CRITICAL API REFERENCE RULE
- **BEFORE writing ANY MLX Swift code**: ALWAYS use the context7 tool to query the exact API signatures and parameter types
- **NEVER assume or guess** MLX Swift API parameters - context7 provides the authoritative reference
- When encountering compilation errors related to types (e.g., `PaddingOrInt`, `IntOrPair`), immediately use context7 to verify the correct types

#### Core Porting Guidelines  
- When porting models from the mlx-python version, ensure the mlx-swift implementation matches the Python version **as closely as possible**.
- For questions about the mlx-swift API, MUST use the context7 tool for reference.
- When porting a VLM model, place the model implementation in `Libraries/MLXVLM/Models`.
- When porting a VLM model, be sure to understand how `Libraries/MLXVLM/MediaProcessing.swift` and `Libraries/MLXVLM/VLMModelFactory.swift` work together.
- The implementation of `KVCache` can be found in `Libraries/MLXLMCommon/KVCache.swift`.
  - createAttentionMask
  - createCausalMask
  - class RotatingKVCache
  - class QuantizedKVCache
  - class ChunkedKVCache
  - class KVCacheSimple
  - protocol KVCache

- MLXLMCommon contains types and code that is generic across many types
of language models, from LLMs to VLMs:
  - Evaluation
  - KVCache
  - Loading
  - UserInput

- **MLXVLM Model Reference Implementation**: Use `Gemma3.swift` as your primary pattern reference.
  - `Gemma3.swift` provides a fully compatible and idiomatic implementation for the MLXVLM framework.
  - Location: `Libraries/MLXVLM/Models/Gemma3.swift`

#### MLX Swift APIs
- MLXFast
  - public static func scaledDotProductAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?,
        memoryEfficientThreshold: Int? = nil, stream: StreamOrDevice = .default
    ) -> MLXArray
  - public static func rmsNorm(
        _ x: MLXArray, weight: MLXArray, eps: Float, stream: StreamOrDevice = .default
    ) -> MLXArray


#### Local Reference Files
- `/Users/magi/Workspace/mlx-llm/gemma-3n-E2B-4bit`
  - `config.summary.json` — A simplified version of `config.json` provided for structural reference only; not intended for production use.
  - `preprocessor_config.json`
  - `config.json`
  - `tokenizer_config.json`

#### Gemma3n model in Python-Swift
- /Users/magi/Workspace/mlx-llm/mlx-vlm/mlx_vlm/models/gemma3n
  - config.py
  - language.py
  - vision.py
  - audio.py
  - gemma3n.py



### Gemma3n MLX-Swift Porting Guidelines
  1. **Direct Python Reference**: Use `/Users/magi/Workspace/mlx-llm/mlx-vlm/mlx_vlm/models/gemma3n/` as primary reference
  2. **Module-by-Module Porting**:
     - `language.py` → `Gemma3nLanguage.swift`
     - `vision.py` → `Gemma3nVision.swift`
     - `audio.py` → `Gemma3nAudio.swift`
     - `gemma3n.py` → `Gemma3n.swift`
     - `config.py` → `Gemma3nConfiguration.swift`
  3. **Preserve Python Logic**: Keep attention mechanisms, layer structures, and forward pass logic identical
  4. **Critical Guideline**:  
    - In `vision.py`, the `MobileNetV5MultiScaleFusionAdapter` performs `bicubic_interpolate` and `nearest_interpolate` operations.  
    - **For MLX-Swift:** Follow the approach in `Gemma3.swift` (`Gemma3Processor`), and use `MediaProcessing` to perform all interpolation (e.g., resizing, upsampling) during preprocessing (from `CIImage` to `MLXArray`).  
    - **Do not** perform any interpolation within the neural network inference pipeline itself. All image resizing and interpolation must be completed before the data enters the model.
  5. **Shared Module Declaration**
    - Shared modules, such as `Gemma3nRMSNorm`, must be declared as `public` so they can be used across different components (for example, in the audio module).
    - When implementing, always ensure that any module intended for cross-file usage is marked as `public` to guarantee visibility and reusability.
  6. Always use MLX API functions for elementwise comparison, not operators.
    - Note: When performing elementwise comparisons on MLXArray, always use the MLX-provided API functions (such as greaterEqual(a, b), less(a, b), equal(a, b), etc.) rather than Swift's comparison operators (like >=, <=, ==, etc.). This ensures correct tensor computation behavior.


### Configuration Porting Strategy
  - **Direct JSON Mapping**: Use `config.json` values directly through Swift `Codable` protocol, mirroring the structure from Python config.py dataclasses
  - Only provide fallback values (`private let _propertyName: Type?` with computed properties) when specific model variants are missing certain fields in their JSON config files, not for general default value handling
  - Focus on essential runtime parameters rather than complex nested configurations
  - Use `config.summary.json` to understand the configuration structure, as it provides a concise overview. Avoid reading the full `config.json` due to its large size.

### To-Do Management
  - Keep an up-to-date `todo.md` to track progress and outstanding tasks.
  - Implement features incrementally, updating `todo.md` as each step is completed.

### Compilation
  - After making changes, ask the user to compile the updated code and share any error messages for further debugging. Do not run `swift build` yourself.

---

# MLX Python to Swift API Mapping Guide (Gemma3n-specific)

This document provides the mapping between Python MLX APIs actually used in the Gemma3n model and their Swift MLX equivalents.

## Core Array Operations

### Array Creation

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `mx.array(value)` | `MLXArray(value)` | Create array from scalar/array |
| `mx.zeros(shape)` | `zeros(shape, type: .float32)` | Create zeros array |
| `mx.ones(shape)` | `ones(shape, type: .float32)` | Create ones array |
| `mx.arange(start, stop, step)` | `MLXArray(start..<stop)` | Simple range only; no step support |
| `mx.ones_like(x)` | `ones(like: x)` | Array of ones with same shape |

### Mathematical Operations

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `mx.exp(x)` | `exp(x)` | Exponential function |
| `mx.log(x)` | `log(x)` | Natural logarithm |
| `mx.sin(x)` | `sin(x)` | Sine function |
| `mx.cos(x)` | `cos(x)` | Cosine function |
| `mx.rsqrt(x)` | `rsqrt(x)` | Reciprocal square root |
| `mx.maximum(a, b)` | `maximum(a, b)` | Element-wise maximum |
| `mx.clip(x, min, max)` | `clip(x, min: min, max: max)` | Clip values to range |
| `mx.power(x, y)` | `pow(x, y)` | Power function |
| `mx.tanh(x)` | `tanh(x)` | Hyperbolic tangent |

### Trigonometric Functions

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `mx.erfinv(x)` | `erfInverse(x)` | Inverse error function |

### Linear Algebra Operations

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `mx.tril(x, k)` | `tril(x, k: k)` | Lower triangular matrix |
| `mx.matmul(a, b)` | `matmul(a, b)` or `a.matmul(b)` | Matrix multiplication |

### Array Shape Manipulation

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `mx.expand_dims(x, axis)` | `expandedDimensions(x, axis: axis)` | Add dimension |
| `mx.concatenate(arrays, axis)` | `concatenated(arrays, axis: axis)` | Concatenate arrays |
| `mx.stack(arrays, axis)` | `stacked(arrays, axis: axis)` | Stack arrays |
| `mx.pad(x, pad_width)` | `padded(x, widths: pad_width, mode: .constant, value: 0)` | Pad array with mode |
| `mx.broadcast_to(x, shape)` | `broadcast(x, to: shape)` | Broadcast to shape |
| `mx.take_along_axis(x, indices, axis)` | `takeAlong(x, indices, axis: axis)` | Take along axis |

### Reduction Operations

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `mx.sum(x, axis, keepdims)` | `x.sum(axis: axis, keepDims: keepdims)` | Sum reduction |
| `mx.mean(x, axis, keepdims)` | `x.mean(axis: axis, keepDims: keepdims)` | Mean reduction |
| `mx.cumsum(x, axis)` | `x.cumsum(axis: axis)` or `cumsum(x, axis: axis)` | Cumulative sum |
| `mx.std(x, axis, keepdims)` | `std(x, axis: axis, keepDims: keepdims, ddof: 0)` | Standard deviation |

### Logical Operations

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `mx.logical_and(a, b)` | `logicalAnd(a, b)` or `a .&& b` | Logical AND |
| `mx.where(condition, x, y)` | `where(condition, x, y)` | Conditional selection |

### Data Type Operations

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `x.astype(dtype)` | `x.asType(dtype)` | Convert data type |
| `x.dtype` | `x.dtype` | Get data type |
| `mx.bool_` | `.bool` | Boolean type |
| `mx.float32` | `.float32` | 32-bit float |

## Gemma3n-Specific Operations

### Audio Processing

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `mx.logaddexp(x, y)` | `logAddExp(x, y)` | Log-add-exp operation |
| `mx.softmax(x, axis)` | `softmax(x, axis: axis, precise: false)` | Softmax function |

### Vision Processing

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `mx.conv2d(input, weight, stride, padding)` | `conv2d(input, weight, stride: stride, padding: padding)` | 2D convolution |

### Language Processing

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `mx.fast.rms_norm(x, weight, eps)` | `MLXFast.rmsNorm(x, weight: weight, eps: eps)` | Fast RMS normalization |
| `mx.fast.scaled_dot_product_attention(queries, keys, values, scale, mask)` | `MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: mask)` | Fast scaled dot-product attention |

### Compilation

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `@partial(mx.compile, shapeless=True)` | `compile(_:)` | JIT compilation decorator |

## Key Differences

### 1. Namespacing
- **Python**: `mx.function_name()` (note: uses `mx` not `mlx`)
- **Swift**: `functionName()` (free functions) or `MLXArray.method()` (static methods)

### 2. Parameter Names
- **Python**: Often positional parameters
- **Swift**: Uses labeled parameters: `axis:`, `keepDims:`, `stream:`

### 3. keepDims Parameter
Swift MLX uses `keepDims:` (camelCase) instead of Python's `keepdims`:
```swift
// Python: keepdims=True
let result = x.sum(axis: 0, keepDims: true)
```

### 4. Data Types
- **Python**: `mx.float32`, `mx.int32`, `mx.bool_`
- **Swift**: `.float32`, `.int32`, `.bool` (enum cases)

### 5. Property Wrapper Initialization

**Correct initialization pattern for @ModuleInfo and @ParameterInfo:**

```swift
public class MyLayer: Module {
    @ModuleInfo var linear: Linear
    @ParameterInfo(key: "weight") var weight: MLXArray
    
    public init() {
        // Correct: Use wrappedValue for initialization
        self._linear.wrappedValue = Linear(10, 20)
        
        super.init()
        
        // For @ParameterInfo with keys, initialize after super.init()
        self._weight.wrappedValue = zeros([10])
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Usage: Access directly (no wrappedValue needed)
        return linear(x)
    }
}
```

**Why use `wrappedValue`:**
- Ensures proper registration with MLX module system
- Required for weight loading and parameter tracking
- Direct assignment may bypass property wrapper logic

**Timing:**
- `@ModuleInfo` properties: Initialize before `super.init()`
- `@ParameterInfo` with keys: Initialize after `super.init()`

## Common Patterns in Gemma3n Porting

### 1. Padding Operations
```python
# Python
mx.pad(x, pad_widths, constant_values=value)
```
```swift
// Swift
padded(x, widths: pad_widths, mode: .constant, value: value)
```

### 2. Attention Mechanisms
```python
# Python
mx.fast.scaled_dot_product_attention(queries, keys, values, scale, mask)
```
```swift
// Swift
MLXFast.scaledDotProductAttention(
    queries: queries, 
    keys: keys, 
    values: values, 
    scale: scale, 
    mask: mask
)
```

### 3. Normalization
```python
# Python
mx.fast.rms_norm(x, weight, eps)
```
```swift
// Swift
MLXFast.rmsNorm(x, weight: weight, eps: eps)
```

### 4. Compilation Decorators
```python
# Python
@partial(mx.compile, shapeless=True)
def some_function(x):
    return x + 1
```
```swift
// Swift  
func someFunction(_ x: MLXArray) -> MLXArray {
    return x + 1
}
let compiledFunction = compile(someFunction)
```

This mapping guide focuses specifically on MLX APIs used in the Gemma3n model implementation.

---

# MLX Neural Network (nn) Python to Swift Mapping Guide

This document provides the mapping between Python MLX NN modules used in the Gemma3n model and their Swift MLX equivalents based on the actual MLX Swift documentation.

## Base Classes and Module Structure

### Module Base Class

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `class MyModel(nn.Module):` | `class MyModel: Module, UnaryLayer` | Swift uses protocol conformance |
| `super().__init__()` | `super.init()` | Swift initialization |
| `def __call__(self, x):` | `public func callAsFunction(_ x: MLXArray) -> MLXArray` | Swift callable protocol |

**Swift Example:**
```swift
public class MyModel: Module, UnaryLayer {
    @ModuleInfo var linear: Linear
    
    public init(inputDim: Int, outputDim: Int) {
        self._linear.wrappedValue = Linear(inputDim, outputDim)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return linear(x)
    }
}
```

## Core Neural Network Layers

### Linear Layer

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `nn.Linear(input_dims, output_dims, bias=True)` | `Linear(inputDimensions, outputDimensions, bias: true)` | Parameter names differ |

**Swift Usage:**
```swift
// With bias (default)
@ModuleInfo var linear1: Linear = Linear(256, 128, bias: true)

// Without bias  
@ModuleInfo var linear2: Linear = Linear(256, 128, bias: false)
```

### Embedding Layer

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `nn.Embedding(vocab_size, embedding_dim)` | `Embedding(vocabularySize, dimensions)` | Parameter names differ |

**Swift Usage:**
```swift
@ModuleInfo var embedding: Embedding = Embedding(vocabularySize: 50000, dimensions: 768)
```

### RMS Normalization

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `nn.RMSNorm(dims, eps=1e-5)` | `RMSNorm(dimensions, eps: 1e-5)` | Parameter names differ |

**Swift Usage:**
```swift
@ModuleInfo var rmsNorm: RMSNorm = RMSNorm(768, eps: 1e-5)
```

### Convolutional Layers

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)` | `Conv1d(inputChannels, outputChannels, kernelSize, stride: stride, padding: padding, dilation: dilation, groups: groups)` | Named parameters in Swift |
| `nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)` | `Conv2d(inputChannels, outputChannels, kernelSize, stride: stride, padding: padding, dilation: dilation, groups: groups)` | Named parameters in Swift |

**Swift Usage:**
```swift
// 1D Convolution
@ModuleInfo var conv1d: Conv1d = Conv1d(
    inputChannels: 64, 
    outputChannels: 128, 
    kernelSize: 3,
    stride: 1,
    padding: 1
)

// 2D Convolution  
@ModuleInfo var conv2d: Conv2d = Conv2d(
    inputChannels: 3,
    outputChannels: 64, 
    kernelSize: 7,
    stride: 2,
    padding: 3
)
```

### Pooling Layers

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `nn.AvgPool2d(kernel_size, stride, padding)` | `AvgPool2d(kernelSize, stride: stride, padding: padding)` | Named parameters in Swift |

**Swift Usage:**
```swift
@ModuleInfo var avgPool: AvgPool2d = AvgPool2d(kernelSize: 2, stride: 2, padding: 0)
```

### Dropout Layers

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `nn.Dropout(p=0.1)` | `Dropout(p: 0.1)` | Named parameter in Swift |
| `nn.Dropout2d(p=0.1)` | `Dropout2d(p: 0.1)` | Named parameter in Swift |
| `nn.Dropout3d(p=0.1)` | `Dropout3d(p: 0.1)` | Named parameter in Swift |

**Swift Usage:**
```swift
@ModuleInfo var dropout: Dropout = Dropout(p: 0.1)
@ModuleInfo var dropout2d: Dropout2d = Dropout2d(p: 0.2)
```

### Identity Layer

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `nn.Identity()` | `Identity()` | Pass-through layer |

**Swift Usage:**
```swift
@ModuleInfo var identity: Identity = Identity()
```

## Positional Encoding Layers

### RoPE (Rotary Positional Embedding)

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `nn.RoPE(dims, traditional=False, base=10000, scale=1.0)` | `RoPE(dimensions, traditional: false, base: 10000, scale: 1.0)` | Named parameters in Swift |

**Swift Usage:**
```swift
let var rope: RoPE = RoPE(
    dimensions: 64,
    traditional: false,
    base: 10000,
    scale: 1.0
)
```
> **Note:**  
> Positional encoding layers (such as RoPE) do not contain trainable parameters (weights), so you do **not** need to use `@ModuleInfo` in Swift. Simply create and use them directly.
> RMSNoScale does not contain learnable parameters, so you do **not** need to use `@ModuleInfo`. 



## Activation Functions

### Free Functions (Direct Usage)

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `nn.relu(x)` | `relu(x)` | Free function |
| `nn.silu(x)` | `silu(x)` | Free function (also called Swish) |
| `nn.gelu(x)` | `gelu(x)` | Free function |
| `nn.gelu_approx(x)` | `geluApproximate(x)` | Approximate GELU |
| `nn.tanh(x)` | `tanh(x)` | Free function |
| `nn.softplus(x)` | `softplus(x)` | Free function |
| `nn.sigmoid(x)` | `sigmoid(x)` | Free function |
| `nn.glu(x, axis=-1)` | `glu(x, axis: -1)` | Gated Linear Unit |

### Activation Modules (Layer Objects)

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `nn.ReLU()` | `ReLU()` | Module form |
| `nn.SiLU()` | `SiLU()` | Module form |
| `nn.GELU()` | `GELU()` | Module form |
| `nn.Tanh()` | `Tanh()` | Module form |
| `nn.Softplus()` | `Softplus()` | Module form |
| `nn.Sigmoid()` | `Sigmoid()` | Module form |
| `nn.GLU()` | `GLU()` | Module form |

**Swift Usage:**
```swift
// Free functions (direct use)
let activated = silu(x)
let gated = glu(x, axis: -1)

// Module forms
@ModuleInfo var activation: SiLU = SiLU()
let result = activation(input)
```

## Multi-Head Attention

| Python MLX | Swift MLX | Notes |
|------------|-----------|--------|
| `nn.MultiHeadAttention(dims, num_heads, query_input_dims, key_input_dims, value_input_dims, value_dims, value_output_dims, bias)` | `MultiHeadAttention(dimensions, numberOfHeads, queryInputDimensions, keyInputDimensions, valueInputDimensions, valueDimensions, valueOutputDimensions, bias)` | Extensive parameter mapping |

**Swift Usage:**
```swift
@ModuleInfo var attention: MultiHeadAttention = MultiHeadAttention(
    dimensions: 768,
    numberOfHeads: 12,
    bias: false
)
```

## Key Differences in Swift MLX NN

### 1. Module Declaration

**Python:**
```python
class MyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
```

**Swift:**
```swift
public class MyLayer: Module, UnaryLayer {
    @ModuleInfo var linear: Linear
    
    public init() {
        self._linear.wrappedValue = Linear(10, 20)
        super.init()
    }
}
```

### 2. Parameter Access

**Python:**
```python
# Parameters automatically tracked
model.parameters()
```

**Swift:**
```swift
// Use @ModuleInfo for automatic tracking
@ModuleInfo var linear: Linear
// Or @ParameterInfo for raw MLXArray parameters
@ParameterInfo var weight: MLXArray

// Access parameters
let params = model.parameters()
```

### 3. Forward Pass

**Python:**
```python
def __call__(self, x):
    return self.linear(x)
```

**Swift:**
```swift
public func callAsFunction(_ x: MLXArray) -> MLXArray {
    return linear(x)
}
```

### 4. Parameter Naming Conventions

- **Python**: snake_case parameters (`input_dims`, `kernel_size`)
- **Swift**: camelCase parameters (`inputDimensions`, `kernelSize`)

### 5. Named Parameters

Swift MLX extensively uses named parameters for clarity:

**Python:**
```python
nn.Conv2d(64, 128, 3, 2, 1)  # positional
```

**Swift:**
```swift
Conv2d(inputChannels: 64, outputChannels: 128, kernelSize: 3, stride: 2, padding: 1)
```

## Common Patterns in Gemma3n Porting

### 1. Layer Initialization

**Python:**
```python
self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
```

**Swift:**
```swift
@ModuleInfo var qProj: Linear

// In init():
self._qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
```

### 2. Multiple Normalizations

**Python:**
```python
self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

**Swift:**
```swift
@ModuleInfo var inputLayernorm: RMSNorm
@ModuleInfo var postAttentionLayernorm: RMSNorm

// In init():
self._inputLayernorm.wrappedValue = RMSNorm(config.hiddenSize, eps: config.rmsNormEps)
self._postAttentionLayernorm.wrappedValue = RMSNorm(config.hiddenSize, eps: config.rmsNormEps)
```

### 3. Activation Usage

**Python:**
```python
x = nn.gelu_approx(gate_proj)
```

**Swift:**
```swift
let x = geluApproximate(gateProj)
```

### 4. Module Printing and Inspection

**Swift:**
```swift
// Print module structure
print(model)

// Get parameter shapes
let shapes = model.mapParameters { $0.shape }
print(shapes)

// Access only local parameters
let localShapes = model.filterMap(
    filter: Module.filterLocalParameters,
    map: Module.mapParameters { $0.shape }
)
```

This mapping guide focuses specifically on the neural network layers and components used in the Gemma3n model implementation, providing accurate Swift MLX equivalents based on the official documentation.

---

# Gemma3n Model Testing and Debugging Guide

## Quick Start Testing

### 1. Basic Text Generation Test
```bash
./mlx-run llm-tool --model /Users/magi/Workspace/mlx-llm/gemma-3n-E2B-4bit --prompt "Hello, how are you?"
```

### 2. Vision-Language Test (with image)
```bash
./mlx-run llm-tool --model /Users/magi/Workspace/mlx-llm/gemma-3n-E2B-4bit --prompt "Describe this image" --image /path/to/image.jpg
```

### 3. Audio Processing Test (if supported)
```bash
./mlx-run llm-tool --model /Users/magi/Workspace/mlx-llm/gemma-3n-E2B-4bit --prompt "What do you hear?" --audio /path/to/audio.wav
```

## Common Runtime Errors and Solutions

### 1. "Unsupported model type: gemma3n"
**Problem**: Model type not recognized by factory system
**Solutions**:
- Check that `VLMTypeRegistry` includes `"gemma3n"` mapping in `VLMModelFactory.swift`
- Verify `config.json` has correct `model_type: "gemma3n"`
- Ensure all imports are correct in the registry file

### 2. Configuration Loading Errors
**Problem**: Model configuration parsing fails
**Debug steps**:
```bash
# Check configuration file structure
cat /path/to/model/config.json | head -20

# Verify processor configuration exists
ls -la /path/to/model/preprocessor_config.json

# Test with debug mode
./mlx-run --debug llm-tool --model /path/to/model --prompt "test"
```

### 3. Weight Loading Issues
**Problem**: Model weights fail to load properly
**Debug approach**:
- Check if all required weight files exist
- Verify weight tensor shapes match model architecture
- Check for quantization compatibility

### 4. Memory Issues
**Problem**: Out of memory during model loading
**Solutions**:
- Use quantized model (4-bit recommended)
- Reduce context length if possible
- Monitor memory usage during loading

## Debugging Workflow

### Step 1: Verify Model Registration
```bash
# Check if model type is registered
grep -r "gemma3n" Libraries/MLXVLM/VLMModelFactory.swift
```

### Step 2: Test Configuration Loading
```bash
# Check model configuration structure
python3 -c "
import json
with open('/path/to/model/config.json', 'r') as f:
    config = json.load(f)
    print('Model Type:', config.get('model_type'))
    print('Has Vision Config:', 'vision_config' in config)
    print('Has Audio Config:', 'audio_config' in config)
    print('Has Text Config:', 'text_config' in config)
"
```

### Step 3: Enable Debug Mode
```bash
# Run with maximum verbosity
./mlx-run --debug llm-tool --model /path/to/model --prompt "test" 2>&1 | tee debug.log
```

### Step 4: Check Dependencies
Ensure all required files are present:
- `config.json` - Main model configuration
- `preprocessor_config.json` - Input processor configuration  
- `tokenizer.json` - Tokenizer configuration
- Weight files (`.safetensors` or `.npz`)

## Performance Optimization

### For Development/Testing
```bash
# Use debug mode for detailed error messages
./mlx-run --debug llm-tool --model /path/to/model --prompt "test"
```

### For Production
```bash
# Use release mode for optimal performance
./mlx-run --release llm-tool --model /path/to/model --prompt "test"
```

## Troubleshooting Checklist

- [ ] Model directory contains all required files
- [ ] `config.json` has `model_type: "gemma3n"`
- [ ] `preprocessor_config.json` has correct processor class
- [ ] All imports in source files are correct
- [ ] Model classes are properly registered in factory
- [ ] Access levels (public/private) are correctly set
- [ ] Property wrapper initialization follows correct pattern

## Logging and Diagnostics

Add these debug prints to troubleshoot specific issues:

```swift
// In model initialization
print("Loading Gemma3n with config: \(config)")
print("Model type: \(config.modelType)")
print("Text config loaded: \(config.textConfig)")
print("Vision config loaded: \(config.visionConfig)")
print("Audio config loaded: \(config.audioConfig)")
```