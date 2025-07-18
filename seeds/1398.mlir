module {
  func.func @main(%arg0: tensor<14x98x21x14x20xi64>, %arg1: tensor<14x98x1x1x20xi64>, %arg2: tensor<66xf32>, %arg3: tensor<62xi1>, %arg4: tensor<1xi1>) -> (tensor<14x98x21x14x20xi64>, tensor<1xf32>, tensor<66xf32>, tensor<1xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<14x98x21x14x20xi64>, tensor<14x98x1x1x20xi64>) -> tensor<14x98x21x14x20xi64>
    %1 = tosa.sub %0, %0 : (tensor<14x98x21x14x20xi64>, tensor<14x98x21x14x20xi64>) -> tensor<14x98x21x14x20xi64>
    %2 = tosa.abs %1 : (tensor<14x98x21x14x20xi64>) -> tensor<14x98x21x14x20xi64>
    %3 = tosa.rsqrt %arg2 : (tensor<66xf32>) -> tensor<66xf32>
    %4 = tosa.reduce_min %3 {axis = 0 : i32} : (tensor<66xf32>) -> tensor<1xf32>
    %in_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %6 = tosa.logical_or %arg3, %arg4 : (tensor<62xi1>, tensor<1xi1>) -> tensor<62xi1>
    %7 = tosa.exp %3 : (tensor<66xf32>) -> tensor<66xf32>
    %8 = tosa.reduce_any %6 {axis = 0 : i32} : (tensor<62xi1>) -> tensor<1xi1>
    return %2, %5, %7, %8 : tensor<14x98x21x14x20xi64>, tensor<1xf32>, tensor<66xf32>, tensor<1xi1>
  }
}
