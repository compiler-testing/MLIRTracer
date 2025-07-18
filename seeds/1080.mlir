module {
  func.func @main(%arg0: tensor<24x83x41xi64>, %arg1: tensor<41x10x23x93x11xf32>, %arg2: tensor<1x1x1x93x11xf32>) -> (tensor<1x83x1xi64>, tensor<41x10x23x93x11xf32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<24x83x41xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x83x41xi64>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<24x83x41xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<24x83x41xi64>
    %2 = tosa.pow %arg1, %arg2 : (tensor<41x10x23x93x11xf32>, tensor<1x1x1x93x11xf32>) -> tensor<41x10x23x93x11xf32>
    %3 = tosa.bitwise_or %1, %1 : (tensor<24x83x41xi64>, tensor<24x83x41xi64>) -> tensor<24x83x41xi64>
    %4 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<24x83x41xi64>) -> tensor<1x83x41xi64>
    %5 = tosa.reduce_sum %4 {axis = 2 : i32} : (tensor<1x83x41xi64>) -> tensor<1x83x1xi64>
    %6 = tosa.sigmoid %2 : (tensor<41x10x23x93x11xf32>) -> tensor<41x10x23x93x11xf32>
    %7 = tosa.pow %6, %6 : (tensor<41x10x23x93x11xf32>, tensor<41x10x23x93x11xf32>) -> tensor<41x10x23x93x11xf32>
    return %5, %7 : tensor<1x83x1xi64>, tensor<41x10x23x93x11xf32>
  }
}
