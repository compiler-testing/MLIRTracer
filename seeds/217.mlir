module {
  func.func @main(%arg0: tensor<7x75x64xi64>, %arg1: tensor<7x64x30xi64>) -> tensor<7x150x30xi64> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<7x75x64xi64>, tensor<7x64x30xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<7x75x30xi64>
    %1 = tosa.bitwise_xor %0, %0 : (tensor<7x75x30xi64>, tensor<7x75x30xi64>) -> tensor<7x75x30xi64>
    %2 = tosa.sub %1, %1 : (tensor<7x75x30xi64>, tensor<7x75x30xi64>) -> tensor<7x75x30xi64>
    %3 = tosa.concat %2, %2 {axis = 1 : i32} : (tensor<7x75x30xi64>, tensor<7x75x30xi64>) -> tensor<7x150x30xi64>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<7x150x30xi64>, tensor<7x150x30xi64>) -> tensor<7x150x30xi64>
    return %4 : tensor<7x150x30xi64>
  }
}
