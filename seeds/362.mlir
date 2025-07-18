module {
  func.func @main(%arg0: tensor<26xi1>, %arg1: tensor<26xi1>, %arg2: tensor<99x34x22xf32>, %arg3: tensor<99x34x22xf32>) -> (tensor<26xi1>, tensor<99x34x22xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<26xi1>, tensor<26xi1>) -> tensor<26xi1>
    %1 = tosa.minimum %arg2, %arg3 : (tensor<99x34x22xf32>, tensor<99x34x22xf32>) -> tensor<99x34x22xf32>
    %2 = tosa.exp %1 : (tensor<99x34x22xf32>) -> tensor<99x34x22xf32>
    return %0, %2 : tensor<26xi1>, tensor<99x34x22xf32>
  }
}
