module {
  func.func @main(%arg0: tensor<32x18x33xf32>, %arg1: tensor<56x86x96xi64>, %arg2: tensor<1x1x96xi64>, %arg3: tensor<57xi1>) -> (tensor<56x86x96xi64>, tensor<32x18x33xf32>, tensor<12xi1>) {
    %0 = tosa.rsqrt %arg0 : (tensor<32x18x33xf32>) -> tensor<32x18x33xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<56x86x96xi64>, tensor<1x1x96xi64>) -> tensor<56x86x96xi64>
    %2 = tosa.exp %0 : (tensor<32x18x33xf32>) -> tensor<32x18x33xf32>
    %3 = tosa.reciprocal %2 : (tensor<32x18x33xf32>) -> tensor<32x18x33xf32>
    %4 = tosa.reduce_any %arg3 {axis = 0 : i32} : (tensor<57xi1>) -> tensor<1xi1>
    %5 = tosa.logical_not %4 : (tensor<1xi1>) -> tensor<1xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_6_size = tosa.const_shape {values = dense<[ 12 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.slice %5, %s_6_start, %s_6_size : (tensor<1xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<12xi1>
    return %1, %3, %6 : tensor<56x86x96xi64>, tensor<32x18x33xf32>, tensor<12xi1>
  }
}
