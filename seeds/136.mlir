module {
  func.func @main(%arg0: tensor<99x72x52x67x30x38xi32>, %arg1: tensor<99x1x1x67x30x1xi32>, %arg2: tensor<54xf32>) -> (tensor<99x72x52x67x30x38xi32>, tensor<54xf32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<99x72x52x67x30x38xi32>, tensor<99x1x1x67x30x1xi32>) -> tensor<99x72x52x67x30x38xi32>
    %1 = tosa.ceil %arg2 : (tensor<54xf32>) -> tensor<54xf32>
    %2 = tosa.reciprocal %1 : (tensor<54xf32>) -> tensor<54xf32>
    %3 = tosa.logical_right_shift %0, %0 : (tensor<99x72x52x67x30x38xi32>, tensor<99x72x52x67x30x38xi32>) -> tensor<99x72x52x67x30x38xi32>
    %4 = tosa.identity %2 : (tensor<54xf32>) -> tensor<54xf32>
    return %3, %4 : tensor<99x72x52x67x30x38xi32>, tensor<54xf32>
  }
}
