module {
  func.func @main(%arg0: tensor<11x6x90xi32>) -> tensor<4x6x2xi32> {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<11x6x90xi32>) -> tensor<1x6x90xi32>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<1x6x90xi32>, tensor<1x6x90xi32>) -> tensor<1x6x90xi32>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<1x6x90xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x6x90xi32>
    %3 = tosa.reduce_sum %2 {axis = 2 : i32} : (tensor<1x6x90xi32>) -> tensor<1x6x1xi32>
    %4 = tosa.concat %3, %3 {axis = 0 : i32} : (tensor<1x6x1xi32>, tensor<1x6x1xi32>) -> tensor<2x6x1xi32>
    %t_5 = tosa.const_shape {values = dense<[ 1, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.tile %4, %t_5 : (tensor<2x6x1xi32>, !tosa.shape<3>) -> tensor<2x6x2xi32>
    %6 = tosa.concat %5, %5 {axis = 0 : i32} : (tensor<2x6x2xi32>, tensor<2x6x2xi32>) -> tensor<4x6x2xi32>
    return %6 : tensor<4x6x2xi32>
  }
}
