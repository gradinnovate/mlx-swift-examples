# MLX Swift Examples - Gemma-3n Fork

This is a fork of the [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples) repository, specifically enhanced and tested with the **Gemma-3n** vision-language model.

## 🎯 Focus: Gemma-3n Vision-Language Model

This fork has been extensively tested and optimized for the [`mlx-community/gemma-3n-E2B-4bit`](https://huggingface.co/mlx-community/gemma-3n-E2B-4bit) model, featuring:

- ✅ **Complete Gemma-3n Implementation**: Full support for multimodal (text + vision + audio) capabilities
- ✅ **Performance Optimizations**: Optimized vision processing pipeline with vectorized operations
- ✅ **Bug Fixes**: Resolved critical issues including `<end_of_turn>` token handling and module initialization
- ✅ **Enhanced VLM Support**: Improved llm-tool with automatic VLM detection and multimodal processing

## 🚀 Quick Start with Gemma-3n

### Text Generation
```bash
./mlx-run llm-tool --model mlx-community/gemma-3n-E2B-4bit --prompt "Hello, how are you?" --vlm
```

### Vision-Language Generation
```bash
./mlx-run llm-tool --model mlx-community/gemma-3n-E2B-4bit --prompt "Describe this image" --image /path/to/image.jpg
```

### Auto-VLM Mode (No --image flag needed)
```bash
./mlx-run llm-tool --model mlx-community/gemma-3n-E2B-4bit --vlm --prompt "What do you see?"
# Then provide image when prompted
```

## 🔧 Key Improvements in This Fork

### 1. Gemma-3n Specific Fixes
- **Module Initialization**: Fixed missing `super.init()` calls in 35+ Module classes
- **DepthwiseConv2d**: Custom implementation to handle MLX Swift Conv2d groups parameter limitations
- **Token Handling**: Automatic `<end_of_turn>` token configuration from model's `eos_token_id`
- **Image Normalization**: Corrected from ImageNet standard to Gemma-3n standard ([0.5, 0.5, 0.5])

### 2. Performance Optimizations
- **Vision Processing**: Vectorized `nearestInterpolate` function (300M→vectorized operations)
- **MLX Swift API**: Corrected usage of `repeated()`, `take()`, `stacked()`, and `broadcast()` functions
- **Multi-scale Fusion**: Implemented tensor-level interpolation matching Python behavior

### 3. Enhanced Developer Experience
- **Debug Output**: Comprehensive logging throughout the processing pipeline
- **Model Structure**: Added `printModelStructure()` function for debugging
- **Auto-detection**: Automatic VLM model detection and configuration

## 📚 Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxvlm) -- vision language model implementations
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxllm) -- large language model implementations

## 🛠️ Available Tools & Applications

### Command Line Tools
- **[llm-tool](Tools/llm-tool/README.md)**: Enhanced with VLM support and Gemma-3n optimizations
- **[image-tool](Tools/image-tool/README.md)**: Generate images using stable diffusion models  
- **[mnist-tool](Tools/mnist-tool/README.md)**: Train LeNet on MNIST dataset
- **[ExampleLLM](Tools/ExampleLLM/README.md)**: Simplified API for LLM interaction

### Applications
- **[VLMEval](Applications/VLMEval/README.md)**: iOS/macOS/visionOS VLM evaluation app
- **[LLMEval](Applications/LLMEval/README.md)**: iOS/macOS LLM evaluation app
- **[MLXChatExample](Applications/MLXChatExample/README.md)**: Chat app supporting both LLMs and VLMs
- **[StableDiffusionExample](Applications/StableDiffusionExample/README.md)**: Image generation app

## 🔬 Tested Model

This fork has been specifically tested and optimized for:
- **Model**: [`mlx-community/gemma-3n-E2B-4bit`](https://huggingface.co/mlx-community/gemma-3n-E2B-4bit)
- **Capabilities**: Text generation, image understanding, multimodal conversations
- **Performance**: Optimized for Apple Silicon with MLX framework

## 💻 Development & Testing

### Building and Running
```bash
# Build and run with debug output
./mlx-run --debug llm-tool --model mlx-community/gemma-3n-E2B-4bit --prompt "test"

# Build and run in release mode for best performance  
./mlx-run --release llm-tool --model mlx-community/gemma-3n-E2B-4bit --prompt "test"

# List available build schemes
./mlx-run --list
```

### Code Formatting
```bash
# Format all code
swift-format format --in-place --recursive Libraries Tools Applications

# Run pre-commit hooks
pre-commit run --all-files
```

## 📦 Installation

Add this fork as a dependency to your Package.swift:

```swift
.package(url: "https://github.com/your-username/mlx-swift-examples", branch: "main"),
```

Then add the libraries you need:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXVLM", package: "mlx-swift-examples"),
        .product(name: "MLXLMCommon", package: "mlx-swift-examples")
    ]),
```

## 🙏 Support

If you find this fork helpful for your Gemma-3n development, consider supporting the work:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow.svg)](https://buymeacoffee.com/gradinnovate)

## 📄 License

This project maintains the same license as the original MLX Swift Examples repository.

## 🔗 Related Links

- **Original Repository**: [ml-explore/mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples)
- **MLX Swift**: [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift)
- **Gemma-3n Model**: [mlx-community/gemma-3n-E2B-4bit](https://huggingface.co/mlx-community/gemma-3n-E2B-4bit)
- **MLX Framework**: [ml-explore/mlx](https://github.com/ml-explore/mlx)