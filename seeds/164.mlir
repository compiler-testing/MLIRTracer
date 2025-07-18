module {
  func.func @main(%arg0: tensor<70x35x67x9x37x14xf32>, %arg1: tensor<57x15x20x27x56xi64>, %arg2: tensor<57x1x20x27x1xi64>) -> (tensor<70x35x67x9x37x14xf32>, tensor<57x15x20x27x56xi64>) {
    %0 = tosa.rsqrt %arg0 : (tensor<70x35x67x9x37x14xf32>) -> tensor<70x35x67x9x37x14xf32>
    %1 = tosa.clamp %0 {min_val = -4.900000e+01 : f32, max_val = 9.400000e+01 : f32} : (tensor<70x35x67x9x37x14xf32>) -> tensor<70x35x67x9x37x14xf32>
    %2 = tosa.logical_right_shift %arg1, %arg2 : (tensor<57x15x20x27x56xi64>, tensor<57x1x20x27x1xi64>) -> tensor<57x15x20x27x56xi64>
    %3 = tosa.bitwise_not %2 : (tensor<57x15x20x27x56xi64>) -> tensor<57x15x20x27x56xi64>
    return %1, %3 : tensor<70x35x67x9x37x14xf32>, tensor<57x15x20x27x56xi64>
  }
}
