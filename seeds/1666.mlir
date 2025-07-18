module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<80x66x66xi32>, %arg3: tensor<17x8xf32>) -> (tensor<i1>, tensor<3x8x11xi32>, tensor<17x8xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    %s_3_start = tosa.const_shape {values = dense<[ 21, 7, 25 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_3_size = tosa.const_shape {values = dense<[ 3, 8, 11 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.slice %arg2, %s_3_start, %s_3_size : (tensor<80x66x66xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<3x8x11xi32>
    %4 = tosa.logical_and %2, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.logical_xor %4, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %6 = tosa.minimum %3, %3 : (tensor<3x8x11xi32>, tensor<3x8x11xi32>) -> tensor<3x8x11xi32>
    %7 = tosa.logical_right_shift %3, %6 : (tensor<3x8x11xi32>, tensor<3x8x11xi32>) -> tensor<3x8x11xi32>
    %8 = tosa.ceil %arg3 : (tensor<17x8xf32>) -> tensor<17x8xf32>
    return %5, %7, %8 : tensor<i1>, tensor<3x8x11xi32>, tensor<17x8xf32>
  }
}
