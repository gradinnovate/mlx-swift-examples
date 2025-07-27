// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Download the model using the `HubApi`.
///
/// This will download `*.safetensors` and `*.json` if the ``ModelConfiguration``
/// represents a Hub id, e.g. `mlx-community/gemma-2-2b-it-4bit`.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``
///
/// - Parameters:
///   - hub: HubApi instance
///   - configuration: the model identifier
///   - progressHandler: callback for progress
/// - Returns: URL for the directory containing downloaded files
public func downloadModel(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> URL {
    do {
        switch configuration.id {
        case .id(let id, let revision):
            // download the model weights
            let repo = Hub.Repo(id: id)
            let modelFiles = ["*.safetensors", "*.json"]
            return try await hub.snapshot(
                from: repo,
                revision: revision,
                matching: modelFiles,
                progressHandler: progressHandler
            )
        case .directory(let directory):
            return directory
        }

    } catch Hub.HubClientError.authorizationRequired {
        // an authorizationRequired means (typically) that the named repo doesn't exist on
        // on the server so retry with local only configuration
        return configuration.modelDirectory(hub: hub)

    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            // Error Domain=NSURLErrorDomain Code=-1009 "The Internet connection appears to be offline."
            // fall back to the local directory
            return configuration.modelDirectory(hub: hub)
        } else {
            throw error
        }
    }
}

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:)``, applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }
    //printModelStructure(model: model)
    // per-model cleanup
    weights = model.sanitize(weights: weights)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}

/// Print the structure of a model, showing module names and their ModuleInfo/ParameterInfo keys.
///
/// This function traverses the model hierarchy and prints each module's path
/// along with the @ModuleInfo and @ParameterInfo keys defined in that module.
///
/// - Parameter model: The model to inspect
public func printModelStructure(model: LanguageModel) {
    print("=== Model Structure ===")
    
    func printModule(_ module: Module, path: String = "", indent: String = "") {
        let moduleName = String(describing: type(of: module))
        print("\(indent)\(path.isEmpty ? "root" : path): \(moduleName)")
        
        // Get all child modules (includes both @ModuleInfo and regular children)
        let childModules = module.children()
        
        // Get all parameters (includes @ParameterInfo)
        let parameters = module.parameters()
        
        // Print @ModuleInfo keys (child modules)
        let moduleKeys = childModules.keys.sorted()
        
        for key in moduleKeys {
            if let item = childModules[key] {
                // Handle all possible NestedItem cases
                switch item {
                case .value(let module):
                    let childName = String(describing: type(of: module))
                    print("\(indent)  @ModuleInfo \(key): \(childName)")
                    
                    // Check if this module is array-like (children with numeric keys)
                    let arrayChildren = module.children()
                    let arrayKeys = arrayChildren.keys.sorted()
                    let hasNumericKeys = !arrayKeys.isEmpty && arrayKeys.allSatisfy { Int($0) != nil }
                    
                    if hasNumericKeys {
                        // Sort numeric keys properly
                        let numericKeys = arrayKeys.compactMap { Int($0) }.sorted().map { String($0) }
                        let displayCount = min(numericKeys.count, 3)
                        
                        for i in 0..<displayCount {
                            let arrayKey = numericKeys[i]
                            if let arrayItem = arrayChildren[arrayKey], case .value(let arrayElement) = arrayItem {
                                let elementName = String(describing: type(of: arrayElement))
                                print("\(indent)    [\(arrayKey)]: \(elementName)")
                            }
                        }
                        
                        if numericKeys.count > 3 {
                            print("\(indent)    ... (\(numericKeys.count - 3) more elements)")
                        }
                    }
                default:
                    // Check if it's a direct array using reflection
                    let mirror = Mirror(reflecting: item)
                    
                    // Try multiple ways to extract array data
                    var arrayValue: [Module]? = nil
                    
                    // Method 1: Direct cast
                    if let directArray = item as? [Module] {
                        arrayValue = directArray
                    }
                    // Method 2: Check if it's wrapped in another type
                    else if let firstChild = mirror.children.first?.value as? [Module] {
                        arrayValue = firstChild
                    }
                    // Method 3: Try to find array in all children
                    else {
                        for child in mirror.children {
                            if let childArray = child.value as? [Module] {
                                arrayValue = childArray
                                break
                            }
                        }
                    }
                    
                    if let arrayValue = arrayValue, !arrayValue.isEmpty {
                        let firstModule = arrayValue[0]
                        let elementType = String(describing: type(of: firstModule))
                        print("\(indent)  @ModuleInfo \(key): Array<\(elementType)>")
                        
                        let displayCount = min(arrayValue.count, 3)
                        for i in 0..<displayCount {
                            let elementName = String(describing: type(of: arrayValue[i]))
                            print("\(indent)    [\(i)]: \(elementName)")
                        }
                        
                        if arrayValue.count > 3 {
                            print("\(indent)    ... (\(arrayValue.count - 3) more elements)")
                        }
                    } else {
                        // Fallback: Check if it's a NestedItem containing arrays
                        let itemType = String(describing: type(of: item))
                        print("\(indent)  @ModuleInfo \(key): \(itemType)")
                        
                        // If it's a NestedItem, try to get its children
                        if itemType.contains("NestedItem") {
                            // Try to cast to a Module-like object and get its children
                            if let moduleItem = item as? Module {
                                let arrayChildren = moduleItem.children()
                                let arrayKeys = arrayChildren.keys.sorted()
                                let hasNumericKeys = !arrayKeys.isEmpty && arrayKeys.allSatisfy { Int($0) != nil }
                                
                                if hasNumericKeys {
                                    let numericKeys = arrayKeys.compactMap { Int($0) }.sorted().map { String($0) }
                                    let displayCount = min(numericKeys.count, 3)
                                    
                                    for i in 0..<displayCount {
                                        let arrayKey = numericKeys[i]
                                        if let arrayItem = arrayChildren[arrayKey], case .value(let arrayElement) = arrayItem {
                                            let elementName = String(describing: type(of: arrayElement))
                                            print("\(indent)    [\(arrayKey)]: \(elementName)")
                                        }
                                    }
                                    
                                    if numericKeys.count > 3 {
                                        print("\(indent)    ... (\(numericKeys.count - 3) more elements)")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Print @ParameterInfo keys (parameters at this level)
        for (paramPath, paramItem) in parameters.sorted(by: { $0.key < $1.key }) {
            if case .value(let param) = paramItem {
                // Check if this parameter belongs directly to the current module
                if path.isEmpty {
                    // Root level - show parameters that don't have dots
                    if !paramPath.contains(".") {
                        print("\(indent)  @ParameterInfo \(paramPath): \(param.shape)")
                    }
                } else {
                    // Check if parameter starts with current path and has exactly one more level
                    if paramPath.hasPrefix(path + ".") {
                        let remainingPath = String(paramPath.dropFirst(path.count + 1))
                        if !remainingPath.contains(".") {
                            print("\(indent)  @ParameterInfo \(remainingPath): \(param.shape)")
                        }
                    }
                }
            }
        }
        
        // Recursively traverse child modules
        for (childPath, childItem) in childModules.sorted(by: { $0.key < $1.key }) {
            if case .value(let childModule) = childItem {
                let fullPath = path.isEmpty ? childPath : "\(path).\(childPath)"
                printModule(childModule, path: fullPath, indent: indent + "  ")
            }
        }
    }
    
    printModule(model)
    print("=== End Structure ===")
}
