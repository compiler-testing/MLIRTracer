module {
  func.func @main(%arg0: tensor<12x23xi32>, %arg1: tensor<1x23xi32>, %arg2: tensor<f32>) -> (tensor<12x23xi32>, tensor<f32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<12x23xi32>, tensor<1x23xi32>) -> tensor<12x23xi32>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<12x23xi32>, tensor<12x23xi32>) -> tensor<12x23xi32>
    %2 = tosa.ceil %arg2 : (tensor<f32>) -> tensor<f32>
    return %1, %2 : tensor<12x23xi32>, tensor<f32>
  }
}
