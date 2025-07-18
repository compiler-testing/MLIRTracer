module {
  func.func @main(%arg0: tensor<82x59x40xi8>) -> tensor<82x1x1xi8> {
    %0 = tosa.clamp %arg0 {min_val = 46 : i8, max_val = 117 : i8} : (tensor<82x59x40xi8>) -> tensor<82x59x40xi8>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<82x59x40xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<82x59x40xi8>
    %2 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<82x59x40xi8>) -> tensor<82x1x40xi8>
    %3 = tosa.bitwise_not %2 : (tensor<82x1x40xi8>) -> tensor<82x1x40xi8>
    %4 = tosa.reduce_sum %3 {axis = 2 : i32} : (tensor<82x1x40xi8>) -> tensor<82x1x1xi8>
    %5 = tosa.abs %4 : (tensor<82x1x1xi8>) -> tensor<82x1x1xi8>
    return %5 : tensor<82x1x1xi8>
  }
}
