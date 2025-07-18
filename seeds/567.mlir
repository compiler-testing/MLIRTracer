module {
  func.func @main(%arg0: tensor<15xi32>, %arg1: tensor<i1>, %arg2: tensor<i1>, %arg3: tensor<79x57x80xi1>) -> (tensor<3xi32>, tensor<1x57x80xi1>, tensor<i1>) {
    %t_0 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<15xi32>, !tosa.shape<1>) -> tensor<15xi32>
    %1 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0>} : (tensor<15xi32>) -> tensor<15xi32>
    %3 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<15xi32>) -> tensor<1xi32>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = tosa.logical_or %arg1, %arg2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %t_6 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.tile %4, %t_6 : (tensor<1xi32>, !tosa.shape<1>) -> tensor<3xi32>
    %7 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<79x57x80xi1>) -> tensor<1x57x80xi1>
    %8 = tosa.logical_not %5 : (tensor<i1>) -> tensor<i1>
    return %6, %7, %8 : tensor<3xi32>, tensor<1x57x80xi1>, tensor<i1>
  }
}
