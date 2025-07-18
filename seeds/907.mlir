module {
  func.func @main(%arg0: tensor<15x56x23xi64>, %arg1: tensor<15x23x27xi64>, %arg2: tensor<83x97x73xi1>) -> (tensor<1x56x27xi64>, tensor<1x73x97xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<15x56x23xi64>, tensor<15x23x27xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<15x56x27xi64>
    %1 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<15x56x27xi64>) -> tensor<1x56x27xi64>
    %2 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<83x97x73xi1>) -> tensor<1x97x73xi1>
    %3 = tosa.concat %2, %2 {axis = 0 : i32} : (tensor<1x97x73xi1>, tensor<1x97x73xi1>) -> tensor<2x97x73xi1>
    %4 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %5 = tosa.transpose %3 {perms = array<i32: 0, 2, 1>} : (tensor<2x97x73xi1>) -> tensor<2x73x97xi1>
    %6 = tosa.reduce_max %5 {axis = 0 : i32} : (tensor<2x73x97xi1>) -> tensor<1x73x97xi1>
    return %1, %6 : tensor<1x56x27xi64>, tensor<1x73x97xi1>
  }
}
