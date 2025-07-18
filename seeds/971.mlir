module {
  func.func @main(%arg0: tensor<65xf32>) -> tensor<1xf32> {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<65xf32>) -> tensor<1xf32>
    %1 = tosa.tanh %0 : (tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.clamp %1 {min_val = -5.800000e+01 : f32, max_val = 1.470000e+02 : f32} : (tensor<1xf32>) -> tensor<1xf32>
    return %2 : tensor<1xf32>
  }
}
