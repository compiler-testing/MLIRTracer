module {
  func.func @main(%arg0: tensor<48x20xf32>, %arg1: tensor<20x37x45x66x27x11xi64>, %arg2: tensor<20x37x45x1x27x1xi64>) -> (tensor<48x20xf32>, tensor<20x37x45x66x27x11xi64>) {
    %0 = tosa.clamp %arg0 {min_val = 3.800000e+01 : f32, max_val = 6.000000e+01 : f32} : (tensor<48x20xf32>) -> tensor<48x20xf32>
    %1 = tosa.exp %0 : (tensor<48x20xf32>) -> tensor<48x20xf32>
    %2 = tosa.bitwise_xor %arg1, %arg2 : (tensor<20x37x45x66x27x11xi64>, tensor<20x37x45x1x27x1xi64>) -> tensor<20x37x45x66x27x11xi64>
    return %1, %2 : tensor<48x20xf32>, tensor<20x37x45x66x27x11xi64>
  }
}
