module {
  func.func @main(%arg0: tensor<84x60x62x39x19x41xf32>, %arg1: tensor<55x52x93x60xi8>) -> (tensor<4x3x8x11x7x12xf32>, tensor<55x52x93x60xi8>) {
    %0 = tosa.exp %arg0 : (tensor<84x60x62x39x19x41xf32>) -> tensor<84x60x62x39x19x41xf32>
    %1 = tosa.exp %0 : (tensor<84x60x62x39x19x41xf32>) -> tensor<84x60x62x39x19x41xf32>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<84x60x62x39x19x41xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<84x60x62x39x19x41xf32>
    %s_3_start = tosa.const_shape {values = dense<[ 80, 34, 54, 21, 12, 16 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_3_size = tosa.const_shape {values = dense<[ 4, 3, 8, 11, 7, 12 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<84x60x62x39x19x41xf32>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<4x3x8x11x7x12xf32>
    %4 = tosa.bitwise_not %arg1 : (tensor<55x52x93x60xi8>) -> tensor<55x52x93x60xi8>
    %5 = tosa.logical_right_shift %4, %4 : (tensor<55x52x93x60xi8>, tensor<55x52x93x60xi8>) -> tensor<55x52x93x60xi8>
    return %3, %5 : tensor<4x3x8x11x7x12xf32>, tensor<55x52x93x60xi8>
  }
}
