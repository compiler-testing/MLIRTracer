module {
  func.func @main(%arg0: tensor<57x30x83xf32>, %arg1: tensor<57x83x60xf32>) -> tensor<57x30x60xi1> {
    %a_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<57x30x83xf32>, tensor<57x83x60xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<57x30x60xf32>
    %1 = tosa.equal %0, %0 : (tensor<57x30x60xf32>, tensor<57x30x60xf32>) -> tensor<57x30x60xi1>
    %2 = tosa.abs %1 : (tensor<57x30x60xi1>) -> tensor<57x30x60xi1>
    return %2 : tensor<57x30x60xi1>
  }
}
