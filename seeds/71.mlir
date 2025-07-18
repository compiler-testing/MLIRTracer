module {
  func.func @main(%arg0: tensor<14x11x54xi64>) -> tensor<1x54xi32> {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<14x11x54xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<14x11x54xi64>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<14x11x54xi64>) -> tensor<11x54xi32>
    %2 = tosa.bitwise_or %1, %1 : (tensor<11x54xi32>, tensor<11x54xi32>) -> tensor<11x54xi32>
    %3 = tosa.identity %2 : (tensor<11x54xi32>) -> tensor<11x54xi32>
    %4 = tosa.bitwise_and %3, %1 : (tensor<11x54xi32>, tensor<11x54xi32>) -> tensor<11x54xi32>
    %5 = tosa.reduce_min %4 {axis = 0 : i32} : (tensor<11x54xi32>) -> tensor<1x54xi32>
    %6 = tosa.minimum %5, %5 : (tensor<1x54xi32>, tensor<1x54xi32>) -> tensor<1x54xi32>
    %7 = tosa.abs %6 : (tensor<1x54xi32>) -> tensor<1x54xi32>
    return %7 : tensor<1x54xi32>
  }
}
