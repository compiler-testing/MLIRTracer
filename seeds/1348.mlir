module {
  func.func @main(%arg0: tensor<90x33x59x91xi32>, %arg1: tensor<f32>) -> (tensor<270x66x177x182xi1>, tensor<f32>) {
    %t_0 = tosa.const_shape {values = dense<[ 3, 2, 3, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<90x33x59x91xi32>, !tosa.shape<4>) -> tensor<270x66x177x182xi32>
    %1 = tosa.greater %0, %0 : (tensor<270x66x177x182xi32>, tensor<270x66x177x182xi32>) -> tensor<270x66x177x182xi1>
    %2 = tosa.rsqrt %arg1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<270x66x177x182xi1>, tensor<270x66x177x182xi1>) -> tensor<270x66x177x182xi1>
    %4 = tosa.rsqrt %2 : (tensor<f32>) -> tensor<f32>
    return %3, %4 : tensor<270x66x177x182xi1>, tensor<f32>
  }
}
