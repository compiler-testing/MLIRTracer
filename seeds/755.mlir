module {
  func.func @main(%arg0: tensor<28x37xi64>, %arg1: tensor<28x37xi64>, %arg2: tensor<48x40x95x77xf32>, %arg3: tensor<76x5x21x59xf32>, %arg4: tensor<76xf32>, %arg5: tensor<39x74x60xi1>) -> (tensor<313728xf32>, tensor<313728xf32>, tensor<48x86x213x76xf32>, tensor<28x37xi64>, tensor<1x74x1xi1>, tensor<313728xf32>, tensor<165x1xi1>, tensor<1x74x1xi1>, tensor<48x86x1x76xi1>, tensor<1x74x1xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<28x37xi64>, tensor<28x37xi64>) -> tensor<28x37xi64>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 48, 86, 213, 76>} : (tensor<48x40x95x77xf32>, tensor<76x5x21x59xf32>, tensor<76xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<48x86x213x76xf32>
    %2 = tosa.reduce_product %1 {axis = 2 : i32} : (tensor<48x86x213x76xf32>) -> tensor<48x86x1x76xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<48x86x1x76xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<48x86x1x76xf32>
    %r_4 = tosa.const_shape {values = dense<[ 313728 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.reshape %3, %r_4 : (tensor<48x86x1x76xf32>, !tosa.shape<1>) -> tensor<313728xf32>
    %5 = tosa.identity %0 : (tensor<28x37xi64>) -> tensor<28x37xi64>
    %6 = tosa.add %4, %4 : (tensor<313728xf32>, tensor<313728xf32>) -> tensor<313728xf32>
    %7 = tosa.clamp %6 {min_val = -2.100000e+01 : f32, max_val = 1.100000e+01 : f32} : (tensor<313728xf32>) -> tensor<313728xf32>
    %8 = tosa.reduce_any %arg5 {axis = 2 : i32} : (tensor<39x74x60xi1>) -> tensor<39x74x1xi1>
    %9 = tosa.arithmetic_right_shift %5, %5 {round = true} : (tensor<28x37xi64>, tensor<28x37xi64>) -> tensor<28x37xi64>
    %10 = tosa.sigmoid %7 : (tensor<313728xf32>) -> tensor<313728xf32>
    %11 = tosa.ceil %10 : (tensor<313728xf32>) -> tensor<313728xf32>
    %12 = tosa.logical_xor %8, %8 : (tensor<39x74x1xi1>, tensor<39x74x1xi1>) -> tensor<39x74x1xi1>
    %13 = tosa.reduce_sum %8 {axis = 0 : i32} : (tensor<39x74x1xi1>) -> tensor<1x74x1xi1>
    %14 = tosa.log %6 : (tensor<313728xf32>) -> tensor<313728xf32>
    %15 = tosa.reduce_min %12 {axis = 1 : i32} : (tensor<39x74x1xi1>) -> tensor<39x1x1xi1>
    %s_16_start = tosa.const_shape {values = dense<[ 1, 0, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_16_size = tosa.const_shape {values = dense<[ 5, 3, 11 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %16 = tosa.slice %15, %s_16_start, %s_16_size : (tensor<39x1x1xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<5x3x11xi1>
    %17 = tosa.logical_right_shift %13, %13 : (tensor<1x74x1xi1>, tensor<1x74x1xi1>) -> tensor<1x74x1xi1>
    %18 = tosa.ceil %1 : (tensor<48x86x213x76xf32>) -> tensor<48x86x213x76xf32>
    %19 = tosa.minimum %5, %9 : (tensor<28x37xi64>, tensor<28x37xi64>) -> tensor<28x37xi64>
    %20 = tosa.bitwise_xor %13, %17 : (tensor<1x74x1xi1>, tensor<1x74x1xi1>) -> tensor<1x74x1xi1>
    %21 = tosa.exp %4 : (tensor<313728xf32>) -> tensor<313728xf32>
    %r_22 = tosa.const_shape {values = dense<[ 165, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %22 = tosa.reshape %16, %r_22 : (tensor<5x3x11xi1>, !tosa.shape<2>) -> tensor<165x1xi1>
    %23 = tosa.bitwise_xor %13, %13 : (tensor<1x74x1xi1>, tensor<1x74x1xi1>) -> tensor<1x74x1xi1>
    %24 = tosa.bitwise_not %13 : (tensor<1x74x1xi1>) -> tensor<1x74x1xi1>
    %25 = tosa.greater_equal %2, %2 : (tensor<48x86x1x76xf32>, tensor<48x86x1x76xf32>) -> tensor<48x86x1x76xi1>
    %26 = tosa.reduce_max %24 {axis = 2 : i32} : (tensor<1x74x1xi1>) -> tensor<1x74x1xi1>
    return %11, %14, %18, %19, %20, %21, %22, %23, %25, %26 : tensor<313728xf32>, tensor<313728xf32>, tensor<48x86x213x76xf32>, tensor<28x37xi64>, tensor<1x74x1xi1>, tensor<313728xf32>, tensor<165x1xi1>, tensor<1x74x1xi1>, tensor<48x86x1x76xi1>, tensor<1x74x1xi1>
  }
}
