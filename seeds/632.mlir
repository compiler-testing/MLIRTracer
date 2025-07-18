module {
  func.func @main(%arg0: tensor<10x99x97xi32>, %arg1: tensor<10x97x74xi32>) -> tensor<10x99x74xi32> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<10x99x97xi32>, tensor<10x97x74xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<10x99x74xi32>
    return %0 : tensor<10x99x74xi32>
  }
}
