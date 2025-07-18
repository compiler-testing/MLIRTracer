module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<79xi8>, %arg2: tensor<79xi8>, %arg3: tensor<47x99x3xi1>, %arg4: tensor<47x1x3xi1>) -> (tensor<i1>, tensor<47x99x3xi1>, tensor<79xi8>) {
    %0 = tosa.clamp %arg0 {min_val = 3.100000e+01 : f32, max_val = 1.150000e+02 : f32} : (tensor<f32>) -> tensor<f32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<79xi8>, tensor<79xi8>) -> tensor<79xi8>
    %2 = tosa.logical_or %arg3, %arg4 : (tensor<47x99x3xi1>, tensor<47x1x3xi1>) -> tensor<47x99x3xi1>
    %3 = tosa.greater_equal %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = tosa.bitwise_and %2, %2 : (tensor<47x99x3xi1>, tensor<47x99x3xi1>) -> tensor<47x99x3xi1>
    %5 = tosa.maximum %1, %1 : (tensor<79xi8>, tensor<79xi8>) -> tensor<79xi8>
    return %3, %4, %5 : tensor<i1>, tensor<47x99x3xi1>, tensor<79xi8>
  }
}
