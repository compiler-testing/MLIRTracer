module {
  func.func @main(%arg0: tensor<98x38x89x59x32x78xf32>, %arg1: tensor<96x38xi64>, %arg2: tensor<23x62x18x58xi1>, %arg3: tensor<23x62x18x58xi1>) -> (tensor<98x38x89x59x32x78xf32>, tensor<192x38xi64>, tensor<1x62x18x58xi1>) {
    %0 = tosa.log %arg0 : (tensor<98x38x89x59x32x78xf32>) -> tensor<98x38x89x59x32x78xf32>
    %t_1 = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %arg1, %t_1 : (tensor<96x38xi64>, !tosa.shape<2>) -> tensor<192x38xi64>
    %2 = tosa.bitwise_and %1, %1 : (tensor<192x38xi64>, tensor<192x38xi64>) -> tensor<192x38xi64>
    %3 = tosa.logical_or %arg2, %arg3 : (tensor<23x62x18x58xi1>, tensor<23x62x18x58xi1>) -> tensor<23x62x18x58xi1>
    %4 = tosa.clamp %2 {min_val = 45 : i64, max_val = 91 : i64} : (tensor<192x38xi64>) -> tensor<192x38xi64>
    %5 = tosa.bitwise_xor %3, %3 : (tensor<23x62x18x58xi1>, tensor<23x62x18x58xi1>) -> tensor<23x62x18x58xi1>
    %6 = tosa.reduce_all %5 {axis = 0 : i32} : (tensor<23x62x18x58xi1>) -> tensor<1x62x18x58xi1>
    return %0, %4, %6 : tensor<98x38x89x59x32x78xf32>, tensor<192x38xi64>, tensor<1x62x18x58xi1>
  }
}
