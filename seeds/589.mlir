module {
  func.func @main(%arg0: tensor<96x46x4x25xi1>, %arg1: tensor<3x25x90xi32>, %arg2: tensor<1x1x1xi32>) -> (tensor<3x25x90xi1>, tensor<96x46x4x25xi1>) {
    %0 = tosa.logical_not %arg0 : (tensor<96x46x4x25xi1>) -> tensor<96x46x4x25xi1>
    %1 = tosa.greater_equal %arg1, %arg2 : (tensor<3x25x90xi32>, tensor<1x1x1xi32>) -> tensor<3x25x90xi1>
    %2 = tosa.logical_xor %0, %0 : (tensor<96x46x4x25xi1>, tensor<96x46x4x25xi1>) -> tensor<96x46x4x25xi1>
    %3 = tosa.logical_right_shift %2, %2 : (tensor<96x46x4x25xi1>, tensor<96x46x4x25xi1>) -> tensor<96x46x4x25xi1>
    return %1, %3 : tensor<3x25x90xi1>, tensor<96x46x4x25xi1>
  }
}
