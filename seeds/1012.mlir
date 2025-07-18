module {
  func.func @main(%arg0: tensor<91x71x69x52x48xf32>, %arg1: tensor<45x50x4x89x39x7xi8>, %arg2: tensor<1x50x1x1x39x7xi8>, %arg3: tensor<52x67x34x14xf32>, %arg4: tensor<27x41x93x10xf32>, %arg5: tensor<27xf32>, %arg6: tensor<i1>, %arg7: tensor<i1>) -> (tensor<91x71x69x52x48xf32>, tensor<52x1x129x27xf32>, tensor<i1>, tensor<45x50x4x89x39x7xi8>) {
    %0 = tosa.rsqrt %arg0 : (tensor<91x71x69x52x48xf32>) -> tensor<91x71x69x52x48xf32>
    %1 = tosa.bitwise_or %arg1, %arg2 : (tensor<45x50x4x89x39x7xi8>, tensor<1x50x1x1x39x7xi8>) -> tensor<45x50x4x89x39x7xi8>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 1, 2, 1, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 52, 176, 129, 27>} : (tensor<52x67x34x14xf32>, tensor<27x41x93x10xf32>, tensor<27xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<52x176x129x27xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.negate %0, %in_zp_3, %out_zp_3 : (tensor<91x71x69x52x48xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<91x71x69x52x48xf32>
    %4 = tosa.reduce_product %2 {axis = 1 : i32} : (tensor<52x176x129x27xf32>) -> tensor<52x1x129x27xf32>
    %5 = tosa.arithmetic_right_shift %1, %1 {round = false} : (tensor<45x50x4x89x39x7xi8>, tensor<45x50x4x89x39x7xi8>) -> tensor<45x50x4x89x39x7xi8>
    %6 = tosa.sub %5, %5 : (tensor<45x50x4x89x39x7xi8>, tensor<45x50x4x89x39x7xi8>) -> tensor<45x50x4x89x39x7xi8>
    %7 = tosa.logical_xor %arg6, %arg7 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %8 = tosa.minimum %6, %5 : (tensor<45x50x4x89x39x7xi8>, tensor<45x50x4x89x39x7xi8>) -> tensor<45x50x4x89x39x7xi8>
    return %3, %4, %7, %8 : tensor<91x71x69x52x48xf32>, tensor<52x1x129x27xf32>, tensor<i1>, tensor<45x50x4x89x39x7xi8>
  }
}
