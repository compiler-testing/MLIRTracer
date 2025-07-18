module {
  func.func @main(%arg0: tensor<10x18x15xi32>, %arg1: tensor<10x15x66xi32>) -> tensor<10x18xi32> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<10x18x15xi32>, tensor<10x15x66xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<10x18x66xi32>
    %1 = tosa.argmax %0 {axis = 2 : i32} : (tensor<10x18x66xi32>) -> tensor<10x18xi32>
    %2 = tosa.bitwise_and %1, %1 : (tensor<10x18xi32>, tensor<10x18xi32>) -> tensor<10x18xi32>
    return %2 : tensor<10x18xi32>
  }
}
