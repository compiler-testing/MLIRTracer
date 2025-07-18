module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<80x18xi32>) -> (tensor<i8>, tensor<80xi32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<i8>, tensor<1xi8>, tensor<1xi8>) -> tensor<i8>
    %1 = tosa.argmax %arg1 {axis = 1 : i32} : (tensor<80x18xi32>) -> tensor<80xi32>
    return %0, %1 : tensor<i8>, tensor<80xi32>
  }
}
