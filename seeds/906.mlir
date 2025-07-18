module {
  func.func @main(%arg0: tensor<42xi1>) -> tensor<1xi1> {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<42xi1>) -> tensor<1xi1>
    %1 = tosa.reduce_any %0 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %2 : tensor<1xi1>
  }
}
