module {
  func.func @main(%arg0: tensor<94xi32>, %arg1: tensor<94xi32>, %arg2: tensor<99x15x35x53x40xf32>, %arg3: tensor<99x1x1x53x1xf32>) -> (tensor<94xi1>, tensor<99x15x35x53x40xf32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<94xi32>, tensor<94xi32>) -> tensor<94xi32>
    %1 = tosa.greater %0, %0 : (tensor<94xi32>, tensor<94xi32>) -> tensor<94xi1>
    %2 = tosa.pow %arg2, %arg3 : (tensor<99x15x35x53x40xf32>, tensor<99x1x1x53x1xf32>) -> tensor<99x15x35x53x40xf32>
    %3 = tosa.reciprocal %2 : (tensor<99x15x35x53x40xf32>) -> tensor<99x15x35x53x40xf32>
    return %1, %3 : tensor<94xi1>, tensor<99x15x35x53x40xf32>
  }
}
