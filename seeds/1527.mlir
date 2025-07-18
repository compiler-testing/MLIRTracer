module {
  func.func @main(%arg0: tensor<23x87x70xi32>, %arg1: tensor<23x70x12xi32>) -> tensor<23x87x12xi32> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<23x87x70xi32>, tensor<23x70x12xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<23x87x12xi32>
    %1 = tosa.intdiv %0, %0 : (tensor<23x87x12xi32>, tensor<23x87x12xi32>) -> tensor<23x87x12xi32>
    return %1 : tensor<23x87x12xi32>
  }
}
