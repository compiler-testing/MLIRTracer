module {
  func.func @main(%arg0: tensor<99x2x7xi1>, %arg1: tensor<99x2x7xi1>) -> tensor<231x1x1x3xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<99x2x7xi1>, tensor<99x2x7xi1>) -> tensor<99x2x7xi1>
    %1 = tosa.argmax %0 {axis = 1 : i32} : (tensor<99x2x7xi1>) -> tensor<99x7xi32>
    %r_2 = tosa.const_shape {values = dense<[ 231, 1, 1, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %1, %r_2 : (tensor<99x7xi32>, !tosa.shape<4>) -> tensor<231x1x1x3xi32>
    %3 = tosa.bitwise_or %2, %2 : (tensor<231x1x1x3xi32>, tensor<231x1x1x3xi32>) -> tensor<231x1x1x3xi32>
    %4 = tosa.minimum %3, %2 : (tensor<231x1x1x3xi32>, tensor<231x1x1x3xi32>) -> tensor<231x1x1x3xi32>
    %5 = tosa.greater_equal %4, %4 : (tensor<231x1x1x3xi32>, tensor<231x1x1x3xi32>) -> tensor<231x1x1x3xi1>
    return %5 : tensor<231x1x1x3xi1>
  }
}
