module {
  func.func @main(%arg0: tensor<96x75x68xi8>, %arg1: tensor<96x1x68xi8>) -> tensor<3x9x11xi8> {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<96x75x68xi8>, tensor<96x1x68xi8>) -> tensor<96x75x68xi8>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<96x75x68xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<96x75x68xi8>
    %2 = tosa.logical_right_shift %1, %0 : (tensor<96x75x68xi8>, tensor<96x75x68xi8>) -> tensor<96x75x68xi8>
    %s_3_start = tosa.const_shape {values = dense<[ 66, 43, 40 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_3_size = tosa.const_shape {values = dense<[ 3, 9, 11 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<96x75x68xi8>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<3x9x11xi8>
    return %3 : tensor<3x9x11xi8>
  }
}
