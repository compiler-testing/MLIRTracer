module {
  func.func @main(%arg0: tensor<64xi64>, %arg1: tensor<84x4x90x45x35x34xf32>) -> (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<84x4x90x45x35x34xf32>, tensor<84x4x90x45x35x34xi1>, tensor<1xi64>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<64xi64>) -> tensor<1xi64>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = tosa.exp %arg1 : (tensor<84x4x90x45x35x34xf32>) -> tensor<84x4x90x45x35x34xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %4 = tosa.floor %2 : (tensor<84x4x90x45x35x34xf32>) -> tensor<84x4x90x45x35x34xf32>
    %5 = tosa.reverse %1 {axis = 0 : i32} : (tensor<1xi64>) -> tensor<1xi64>
    %6 = tosa.clz %1 : (tensor<1xi64>) -> tensor<1xi64>
    %7 = tosa.greater_equal %2, %4 : (tensor<84x4x90x45x35x34xf32>, tensor<84x4x90x45x35x34xf32>) -> tensor<84x4x90x45x35x34xi1>
    %8 = tosa.logical_xor %7, %7 : (tensor<84x4x90x45x35x34xi1>, tensor<84x4x90x45x35x34xi1>) -> tensor<84x4x90x45x35x34xi1>
    %9 = tosa.logical_or %7, %8 : (tensor<84x4x90x45x35x34xi1>, tensor<84x4x90x45x35x34xi1>) -> tensor<84x4x90x45x35x34xi1>
    %10 = tosa.sigmoid %4 : (tensor<84x4x90x45x35x34xf32>) -> tensor<84x4x90x45x35x34xf32>
    %in_zp_11 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_11 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %11 = tosa.negate %9, %in_zp_11, %out_zp_11 : (tensor<84x4x90x45x35x34xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<84x4x90x45x35x34xi1>
    %12 = tosa.bitwise_and %11, %11 : (tensor<84x4x90x45x35x34xi1>, tensor<84x4x90x45x35x34xi1>) -> tensor<84x4x90x45x35x34xi1>
    %13 = tosa.reverse %0 {axis = 0 : i32} : (tensor<1xi64>) -> tensor<1xi64>
    return %3, %5, %6, %10, %12, %13 : tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<84x4x90x45x35x34xf32>, tensor<84x4x90x45x35x34xi1>, tensor<1xi64>
  }
}
