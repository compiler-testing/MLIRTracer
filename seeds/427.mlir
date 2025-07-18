module {
  func.func @main(%arg0: tensor<51x58x2xi1>, %arg1: tensor<51x2x62xi1>) -> tensor<51x58x1xi1> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<51x58x2xi1>, tensor<51x2x62xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<51x58x62xi1>
    %1 = tosa.reduce_sum %0 {axis = 2 : i32} : (tensor<51x58x62xi1>) -> tensor<51x58x1xi1>
    return %1 : tensor<51x58x1xi1>
  }
}
