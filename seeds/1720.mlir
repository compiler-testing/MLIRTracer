module {
  func.func @main(%arg0: tensor<92xi32>, %arg1: tensor<92xi32>, %arg2: tensor<29x29x95xf32>) -> (tensor<92xi32>, tensor<29x1x1xf32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<92xi32>, tensor<92xi32>) -> tensor<92xi32>
    %1 = tosa.tanh %arg2 : (tensor<29x29x95xf32>) -> tensor<29x29x95xf32>
    %2 = tosa.reduce_min %1 {axis = 2 : i32} : (tensor<29x29x95xf32>) -> tensor<29x29x1xf32>
    %3 = tosa.reduce_sum %2 {axis = 1 : i32} : (tensor<29x29x1xf32>) -> tensor<29x1x1xf32>
    return %0, %3 : tensor<92xi32>, tensor<29x1x1xf32>
  }
}
