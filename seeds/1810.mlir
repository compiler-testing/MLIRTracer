module {
  func.func @main(%arg0: tensor<66x25xi32>, %arg1: tensor<37xf32>) -> (tensor<2x55xi32>, tensor<37xf32>) {
    %r_0 = tosa.const_shape {values = dense<[ 2, 55, 15 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<66x25xi32>, !tosa.shape<3>) -> tensor<2x55x15xi32>
    %1 = tosa.bitwise_or %0, %0 : (tensor<2x55x15xi32>, tensor<2x55x15xi32>) -> tensor<2x55x15xi32>
    %2 = tosa.rsqrt %arg1 : (tensor<37xf32>) -> tensor<37xf32>
    %3 = tosa.argmax %1 {axis = 2 : i32} : (tensor<2x55x15xi32>) -> tensor<2x55xi32>
    %4 = tosa.minimum %2, %2 : (tensor<37xf32>, tensor<37xf32>) -> tensor<37xf32>
    %5 = tosa.arithmetic_right_shift %3, %3 {round = true} : (tensor<2x55xi32>, tensor<2x55xi32>) -> tensor<2x55xi32>
    %6 = tosa.ceil %4 : (tensor<37xf32>) -> tensor<37xf32>
    return %5, %6 : tensor<2x55xi32>, tensor<37xf32>
  }
}
