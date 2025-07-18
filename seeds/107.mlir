module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<17x82x39x97xf32>) -> (tensor<i1>, tensor<17x82x39x97xf32>, tensor<17x82x39x97xf32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %3 = tosa.rsqrt %arg2 : (tensor<17x82x39x97xf32>) -> tensor<17x82x39x97xf32>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %4 = tosa.negate %2, %in_zp_4, %out_zp_4 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %5 = tosa.minimum %3, %3 : (tensor<17x82x39x97xf32>, tensor<17x82x39x97xf32>) -> tensor<17x82x39x97xf32>
    %6 = tosa.ceil %3 : (tensor<17x82x39x97xf32>) -> tensor<17x82x39x97xf32>
    return %4, %5, %6 : tensor<i1>, tensor<17x82x39x97xf32>, tensor<17x82x39x97xf32>
  }
}
