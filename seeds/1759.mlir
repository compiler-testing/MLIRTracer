module {
  func.func @main(%arg0: tensor<26xi1>, %arg1: tensor<1xi1>, %arg2: tensor<54x64x43x7x77x3xi8>, %arg3: tensor<1x1x43x1x1x3xi8>, %arg4: tensor<50x60x36x52x65xf32>) -> (tensor<54x64x43x7x77x3xi8>, tensor<26xi1>, tensor<50x60x36x52x65xi1>, tensor<50x60x36x52x65xi1>, tensor<50x60x36x52x65xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<26xi1>, tensor<1xi1>) -> tensor<26xi1>
    %1 = tosa.minimum %arg2, %arg3 : (tensor<54x64x43x7x77x3xi8>, tensor<1x1x43x1x1x3xi8>) -> tensor<54x64x43x7x77x3xi8>
    %2 = tosa.log %arg4 : (tensor<50x60x36x52x65xf32>) -> tensor<50x60x36x52x65xf32>
    %3 = tosa.bitwise_and %0, %0 : (tensor<26xi1>, tensor<26xi1>) -> tensor<26xi1>
    %4 = tosa.logical_xor %3, %0 : (tensor<26xi1>, tensor<26xi1>) -> tensor<26xi1>
    %r_5 = tosa.const_shape {values = dense<[ 26 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %5 = tosa.reshape %4, %r_5 : (tensor<26xi1>, !tosa.shape<1>) -> tensor<26xi1>
    %6 = tosa.floor %2 : (tensor<50x60x36x52x65xf32>) -> tensor<50x60x36x52x65xf32>
    %7 = tosa.greater %2, %6 : (tensor<50x60x36x52x65xf32>, tensor<50x60x36x52x65xf32>) -> tensor<50x60x36x52x65xi1>
    %8 = tosa.logical_right_shift %7, %7 : (tensor<50x60x36x52x65xi1>, tensor<50x60x36x52x65xi1>) -> tensor<50x60x36x52x65xi1>
    %in_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %9 = tosa.negate %7, %in_zp_9, %out_zp_9 : (tensor<50x60x36x52x65xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<50x60x36x52x65xi1>
    %10 = tosa.ceil %6 : (tensor<50x60x36x52x65xf32>) -> tensor<50x60x36x52x65xf32>
    return %1, %5, %8, %9, %10 : tensor<54x64x43x7x77x3xi8>, tensor<26xi1>, tensor<50x60x36x52x65xi1>, tensor<50x60x36x52x65xi1>, tensor<50x60x36x52x65xf32>
  }
}
