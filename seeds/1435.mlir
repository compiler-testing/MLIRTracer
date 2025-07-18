module {
  func.func @main(%arg0: tensor<61x17x79xi1>, %arg1: tensor<61x79x98xi1>) -> tensor<1x17x98xi1> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<61x17x79xi1>, tensor<61x79x98xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<61x17x98xi1>
    %1 = tosa.logical_or %0, %0 : (tensor<61x17x98xi1>, tensor<61x17x98xi1>) -> tensor<61x17x98xi1>
    %2 = tosa.reduce_all %1 {axis = 0 : i32} : (tensor<61x17x98xi1>) -> tensor<1x17x98xi1>
    return %2 : tensor<1x17x98xi1>
  }
}
