module {
  func.func @main(%arg0: tensor<99x6x55x49xi1>, %arg1: tensor<1x1x55x49xi1>, %arg2: tensor<94x41x82xf32>, %arg3: tensor<1x41x1xf32>) -> (tensor<99x55x49xi32>, tensor<94x41x82xf32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<99x6x55x49xi1>, tensor<1x1x55x49xi1>) -> tensor<99x6x55x49xi1>
    %1 = tosa.argmax %0 {axis = 1 : i32} : (tensor<99x6x55x49xi1>) -> tensor<99x55x49xi32>
    %2 = tosa.intdiv %1, %1 : (tensor<99x55x49xi32>, tensor<99x55x49xi32>) -> tensor<99x55x49xi32>
    %3 = tosa.pow %arg2, %arg3 : (tensor<94x41x82xf32>, tensor<1x41x1xf32>) -> tensor<94x41x82xf32>
    return %2, %3 : tensor<99x55x49xi32>, tensor<94x41x82xf32>
  }
}
