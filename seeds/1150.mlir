module {
  func.func @main(%arg0: tensor<90x79x51x41x11xf32>, %arg1: tensor<91x98x69x16xf32>, %arg2: tensor<13x75x78x58xf32>, %arg3: tensor<13xf32>, %arg4: tensor<65xi8>, %arg5: tensor<65xi8>, %arg6: tensor<94xi1>) -> (tensor<90x79x51x41x11xf32>, tensor<1x175x150x13xf32>, tensor<91x175x150x13xf32>, tensor<65xi8>, tensor<65xi8>, tensor<91x175x150x13xf32>, tensor<1xi1>) {
    %0 = tosa.tanh %arg0 : (tensor<90x79x51x41x11xf32>) -> tensor<90x79x51x41x11xf32>
    %1 = tosa.rsqrt %0 : (tensor<90x79x51x41x11xf32>) -> tensor<90x79x51x41x11xf32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg1, %arg2, %arg3, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 2>, stride = array<i64: 1, 1>, out_shape = array<i64: 91, 175, 150, 13>} : (tensor<91x98x69x16xf32>, tensor<13x75x78x58xf32>, tensor<13xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<91x175x150x13xf32>
    %3 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<91x175x150x13xf32>) -> tensor<1x175x150x13xf32>
    %4 = tosa.bitwise_or %arg4, %arg5 : (tensor<65xi8>, tensor<65xi8>) -> tensor<65xi8>
    %5 = tosa.tanh %2 : (tensor<91x175x150x13xf32>) -> tensor<91x175x150x13xf32>
    %6 = tosa.tanh %5 : (tensor<91x175x150x13xf32>) -> tensor<91x175x150x13xf32>
    %7 = tosa.bitwise_and %4, %4 : (tensor<65xi8>, tensor<65xi8>) -> tensor<65xi8>
    %8 = tosa.minimum %7, %7 : (tensor<65xi8>, tensor<65xi8>) -> tensor<65xi8>
    %9 = tosa.bitwise_or %4, %4 : (tensor<65xi8>, tensor<65xi8>) -> tensor<65xi8>
    %10 = tosa.abs %9 : (tensor<65xi8>) -> tensor<65xi8>
    %11 = tosa.tanh %2 : (tensor<91x175x150x13xf32>) -> tensor<91x175x150x13xf32>
    %12 = tosa.reduce_any %arg6 {axis = 0 : i32} : (tensor<94xi1>) -> tensor<1xi1>
    return %1, %3, %6, %8, %10, %11, %12 : tensor<90x79x51x41x11xf32>, tensor<1x175x150x13xf32>, tensor<91x175x150x13xf32>, tensor<65xi8>, tensor<65xi8>, tensor<91x175x150x13xf32>, tensor<1xi1>
  }
}
