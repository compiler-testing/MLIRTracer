module {
  func.func @main(%arg0: tensor<30x100x56x70x68x99xf32>) -> tensor<30x100x56x70x68x99xf32> {
    %0 = tosa.clamp %arg0 {min_val = -3.900000e+01 : f32, max_val = 1.200000e+02 : f32} : (tensor<30x100x56x70x68x99xf32>) -> tensor<30x100x56x70x68x99xf32>
    return %0 : tensor<30x100x56x70x68x99xf32>
  }
}
