module {
  func.func @main(%arg0: tensor<94xf32>) -> tensor<188xf32> {
    %0 = tosa.log %arg0 : (tensor<94xf32>) -> tensor<94xf32>
    %1 = tosa.clamp %0 {min_val = -9.000000e+00 : f32, max_val = 7.000000e+01 : f32} : (tensor<94xf32>) -> tensor<94xf32>
    %t_2 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.tile %1, %t_2 : (tensor<94xf32>, !tosa.shape<1>) -> tensor<188xf32>
    return %2 : tensor<188xf32>
  }
}
