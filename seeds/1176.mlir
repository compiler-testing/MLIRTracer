module {
  func.func @main(%arg0: tensor<53x100x91x66x24x89xf32>, %arg1: tensor<58xi1>) -> (tensor<53x100x91x66x24x89xf32>, tensor<58xi1>, tensor<1xi1>) {
    %0 = tosa.exp %arg0 : (tensor<53x100x91x66x24x89xf32>) -> tensor<53x100x91x66x24x89xf32>
    %1 = tosa.clamp %0 {min_val = -2.000000e+00 : f32, max_val = 2.200000e+01 : f32} : (tensor<53x100x91x66x24x89xf32>) -> tensor<53x100x91x66x24x89xf32>
    %2 = tosa.exp %1 : (tensor<53x100x91x66x24x89xf32>) -> tensor<53x100x91x66x24x89xf32>
    %3 = tosa.logical_not %arg1 : (tensor<58xi1>) -> tensor<58xi1>
    %4 = tosa.logical_or %3, %3 : (tensor<58xi1>, tensor<58xi1>) -> tensor<58xi1>
    %5 = tosa.reduce_all %3 {axis = 0 : i32} : (tensor<58xi1>) -> tensor<1xi1>
    %6 = tosa.bitwise_xor %5, %5 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %2, %4, %6 : tensor<53x100x91x66x24x89xf32>, tensor<58xi1>, tensor<1xi1>
  }
}
