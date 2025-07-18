module {
  func.func @main(%arg0: tensor<87x92xf32>) -> tensor<87x92xf32> {
    %0 = tosa.clamp %arg0 {min_val = -4.900000e+01 : f32, max_val = 2.500000e+01 : f32} : (tensor<87x92xf32>) -> tensor<87x92xf32>
    return %0 : tensor<87x92xf32>
  }
}
