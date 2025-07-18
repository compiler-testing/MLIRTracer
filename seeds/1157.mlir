module {
  func.func @main(%arg0: tensor<8x99x80xi1>, %arg1: tensor<8x80x14xi1>) -> tensor<8x99x14xi1> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<8x99x80xi1>, tensor<8x80x14xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<8x99x14xi1>
    return %0 : tensor<8x99x14xi1>
  }
}
