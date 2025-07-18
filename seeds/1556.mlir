module {
  func.func @main(%arg0: tensor<32x72x20x31x44x54xi16>, %arg1: tensor<1x72x1x31x1x54xi16>, %arg2: tensor<46x27x93xi8>, %arg3: tensor<1x1x93xi8>, %arg4: tensor<i1>, %arg5: tensor<60x94x61x89xf32>, %arg6: tensor<93x97x30x52xf32>, %arg7: tensor<93xf32>) -> (tensor<46x27x93xi8>, tensor<8x8x12x1x6x10xi16>, tensor<60x285x154x93xf32>, tensor<i1>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<32x72x20x31x44x54xi16>, tensor<1x72x1x31x1x54xi16>) -> tensor<32x72x20x31x44x54xi16>
    %1 = tosa.minimum %arg2, %arg3 : (tensor<46x27x93xi8>, tensor<1x1x93xi8>) -> tensor<46x27x93xi8>
    %2 = tosa.logical_not %arg4 : (tensor<i1>) -> tensor<i1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %arg5, %arg6, %arg7, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 60, 285, 154, 93>} : (tensor<60x94x61x89xf32>, tensor<93x97x30x52xf32>, tensor<93xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<60x285x154x93xf32>
    %s_4_start = tosa.const_shape {values = dense<[ 24, 23, 8, 1, 11, 14 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_4_size = tosa.const_shape {values = dense<[ 8, 8, 12, 1, 6, 10 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %4 = tosa.slice %0, %s_4_start, %s_4_size : (tensor<32x72x20x31x44x54xi16>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<8x8x12x1x6x10xi16>
    %5 = tosa.floor %3 : (tensor<60x285x154x93xf32>) -> tensor<60x285x154x93xf32>
    %6 = tosa.rsqrt %5 : (tensor<60x285x154x93xf32>) -> tensor<60x285x154x93xf32>
    %7 = tosa.abs %2 : (tensor<i1>) -> tensor<i1>
    return %1, %4, %6, %7 : tensor<46x27x93xi8>, tensor<8x8x12x1x6x10xi16>, tensor<60x285x154x93xf32>, tensor<i1>
  }
}
