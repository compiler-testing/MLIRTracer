module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<95x1x60x20xf32>) -> (tensor<f32>, tensor<2x3x12x10xf32>) {
    %0 = tosa.log %arg0 : (tensor<f32>) -> tensor<f32>
    %s_1_start = tosa.const_shape {values = dense<[ 8, 0, 48, 10 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_1_size = tosa.const_shape {values = dense<[ 2, 3, 12, 10 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.slice %arg1, %s_1_start, %s_1_size : (tensor<95x1x60x20xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x3x12x10xf32>
    %2 = tosa.clamp %1 {min_val = -5.700000e+01 : f32, max_val = -8.000000e+00 : f32} : (tensor<2x3x12x10xf32>) -> tensor<2x3x12x10xf32>
    return %0, %2 : tensor<f32>, tensor<2x3x12x10xf32>
  }
}
