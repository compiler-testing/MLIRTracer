module {
  func.func @main(%arg0: tensor<71x36xi32>, %arg1: tensor<71x36xi32>, %arg2: tensor<45x56x13xi1>, %arg3: tensor<45x56x1xi1>, %arg4: tensor<55xf32>) -> (tensor<45x56x13xi1>, tensor<71x36xi32>, tensor<1xi1>, tensor<55xf32>, tensor<55xf32>, tensor<55xi1>, tensor<55xi1>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<71x36xi32>, tensor<71x36xi32>) -> tensor<71x36xi32>
    %1 = tosa.logical_and %arg2, %arg3 : (tensor<45x56x13xi1>, tensor<45x56x1xi1>) -> tensor<45x56x13xi1>
    %2 = tosa.intdiv %0, %0 : (tensor<71x36xi32>, tensor<71x36xi32>) -> tensor<71x36xi32>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<71x36xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<71x36xi32>
    %4 = tosa.bitwise_or %3, %2 : (tensor<71x36xi32>, tensor<71x36xi32>) -> tensor<71x36xi32>
    %5 = tosa.ceil %arg4 : (tensor<55xf32>) -> tensor<55xf32>
    %6 = tosa.sigmoid %5 : (tensor<55xf32>) -> tensor<55xf32>
    %7 = tosa.reciprocal %6 : (tensor<55xf32>) -> tensor<55xf32>
    %8 = tosa.greater %5, %7 : (tensor<55xf32>, tensor<55xf32>) -> tensor<55xi1>
    %9 = tosa.bitwise_xor %8, %8 : (tensor<55xi1>, tensor<55xi1>) -> tensor<55xi1>
    %10 = tosa.reduce_sum %8 {axis = 0 : i32} : (tensor<55xi1>) -> tensor<1xi1>
    %11 = tosa.log %7 : (tensor<55xf32>) -> tensor<55xf32>
    %12 = tosa.exp %7 : (tensor<55xf32>) -> tensor<55xf32>
    %13 = tosa.add %8, %8 : (tensor<55xi1>, tensor<55xi1>) -> tensor<55xi1>
    %14 = tosa.logical_xor %9, %13 : (tensor<55xi1>, tensor<55xi1>) -> tensor<55xi1>
    %15 = tosa.bitwise_xor %8, %9 : (tensor<55xi1>, tensor<55xi1>) -> tensor<55xi1>
    return %1, %4, %10, %11, %12, %14, %15 : tensor<45x56x13xi1>, tensor<71x36xi32>, tensor<1xi1>, tensor<55xf32>, tensor<55xf32>, tensor<55xi1>, tensor<55xi1>
  }
}
