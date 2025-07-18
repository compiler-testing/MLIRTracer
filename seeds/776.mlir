module {
  func.func @main(%arg0: tensor<84x27xi64>, %arg1: tensor<84x27xi64>, %arg2: tensor<62xi1>) -> (tensor<168x54xi64>, tensor<62xi1>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<84x27xi64>, tensor<84x27xi64>) -> tensor<84x27xi64>
    %t_1 = tosa.const_shape {values = dense<[ 2, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %0, %t_1 : (tensor<84x27xi64>, !tosa.shape<2>) -> tensor<168x54xi64>
    %2 = tosa.logical_not %arg2 : (tensor<62xi1>) -> tensor<62xi1>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<168x54xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<168x54xi64>
    %4 = tosa.logical_xor %2, %2 : (tensor<62xi1>, tensor<62xi1>) -> tensor<62xi1>
    return %3, %4 : tensor<168x54xi64>, tensor<62xi1>
  }
}
