module {
  func.func @main(%arg0: tensor<100x9x2x15xf32>) -> tensor<100x1x2x15xf32> {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<100x9x2x15xf32>) -> tensor<100x1x2x15xf32>
    %1 = tosa.exp %0 : (tensor<100x1x2x15xf32>) -> tensor<100x1x2x15xf32>
    %2 = tosa.maximum %1, %1 : (tensor<100x1x2x15xf32>, tensor<100x1x2x15xf32>) -> tensor<100x1x2x15xf32>
    %3 = tosa.clamp %2 {min_val = 5.700000e+01 : f32, max_val = 1.590000e+02 : f32} : (tensor<100x1x2x15xf32>) -> tensor<100x1x2x15xf32>
    %4 = tosa.reduce_product %3 {axis = 1 : i32} : (tensor<100x1x2x15xf32>) -> tensor<100x1x2x15xf32>
    return %4 : tensor<100x1x2x15xf32>
  }
}
