module {
  func.func @main(%arg0: tensor<76x21x62xi32>) -> tensor<1596xi32> {
    %0 = tosa.reduce_sum %arg0 {axis = 2 : i32} : (tensor<76x21x62xi32>) -> tensor<76x21x1xi32>
    %r_1 = tosa.const_shape {values = dense<[ 1596 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.reshape %0, %r_1 : (tensor<76x21x1xi32>, !tosa.shape<1>) -> tensor<1596xi32>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<1596xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1596xi32>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<1596xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1596xi32>
    return %3 : tensor<1596xi32>
  }
}
