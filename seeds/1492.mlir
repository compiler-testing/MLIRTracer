module {
  func.func @main(%arg0: tensor<98x4x77x20x15xf32>) -> tensor<98x4x77x20x15xf32> {
    %0 = tosa.clamp %arg0 {min_val = 6.200000e+01 : f32, max_val = 8.700000e+01 : f32} : (tensor<98x4x77x20x15xf32>) -> tensor<98x4x77x20x15xf32>
    return %0 : tensor<98x4x77x20x15xf32>
  }
}
