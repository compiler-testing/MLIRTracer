module {
  func.func @main(%arg0: tensor<7x86x15x39x86x38xf32>, %arg1: tensor<59x93x73x18xi8>, %arg2: tensor<46x74xi1>, %arg3: tensor<1x74xi1>) -> (tensor<2470x43x3x3612xf32>, tensor<59x93x73x18xi8>, tensor<46x74xi1>, tensor<7x86x15x39x86x38xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<7x86x15x39x86x38xf32>) -> tensor<7x86x15x39x86x38xf32>
    %1 = tosa.reverse %arg1 {axis = 3 : i32} : (tensor<59x93x73x18xi8>) -> tensor<59x93x73x18xi8>
    %r_2 = tosa.const_shape {values = dense<[ 2470, 43, 3, 3612 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %0, %r_2 : (tensor<7x86x15x39x86x38xf32>, !tosa.shape<4>) -> tensor<2470x43x3x3612xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<59x93x73x18xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<59x93x73x18xi8>
    %4 = tosa.logical_left_shift %3, %3 : (tensor<59x93x73x18xi8>, tensor<59x93x73x18xi8>) -> tensor<59x93x73x18xi8>
    %5 = tosa.logical_xor %arg2, %arg3 : (tensor<46x74xi1>, tensor<1x74xi1>) -> tensor<46x74xi1>
    %6 = tosa.logical_xor %5, %5 : (tensor<46x74xi1>, tensor<46x74xi1>) -> tensor<46x74xi1>
    %7 = tosa.sigmoid %0 : (tensor<7x86x15x39x86x38xf32>) -> tensor<7x86x15x39x86x38xf32>
    return %2, %4, %6, %7 : tensor<2470x43x3x3612xf32>, tensor<59x93x73x18xi8>, tensor<46x74xi1>, tensor<7x86x15x39x86x38xf32>
  }
}
