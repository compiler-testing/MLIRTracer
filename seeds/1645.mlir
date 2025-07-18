module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<32x34x21x15xf32>, %arg3: tensor<25x40x79x70xf32>, %arg4: tensor<25xf32>, %arg5: tensor<24x13x84xi1>, %arg6: tensor<1x1x84xi1>) -> (tensor<i16>, tensor<24x13x84xi1>, tensor<32x108x103x25xf32>, tensor<32x108x103x1xi1>, tensor<32x216x309x75xf32>, tensor<32x108x103xi32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 32, 108, 103, 25>} : (tensor<32x34x21x15xf32>, tensor<25x40x79x70xf32>, tensor<25xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<32x108x103x25xf32>
    %2 = tosa.sub %0, %0 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %3 = tosa.bitwise_and %2, %0 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %4 = tosa.logical_xor %arg5, %arg6 : (tensor<24x13x84xi1>, tensor<1x1x84xi1>) -> tensor<24x13x84xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %5 = tosa.negate %3, %in_zp_5, %out_zp_5 : (tensor<i16>, tensor<1xi16>, tensor<1xi16>) -> tensor<i16>
    %6 = tosa.exp %1 : (tensor<32x108x103x25xf32>) -> tensor<32x108x103x25xf32>
    %7 = tosa.logical_left_shift %4, %4 : (tensor<24x13x84xi1>, tensor<24x13x84xi1>) -> tensor<24x13x84xi1>
    %8 = tosa.pow %1, %1 : (tensor<32x108x103x25xf32>, tensor<32x108x103x25xf32>) -> tensor<32x108x103x25xf32>
    %9 = tosa.greater %8, %8 : (tensor<32x108x103x25xf32>, tensor<32x108x103x25xf32>) -> tensor<32x108x103x25xi1>
    %10 = tosa.abs %1 : (tensor<32x108x103x25xf32>) -> tensor<32x108x103x25xf32>
    %11 = tosa.maximum %1, %6 : (tensor<32x108x103x25xf32>, tensor<32x108x103x25xf32>) -> tensor<32x108x103x25xf32>
    %12 = tosa.logical_left_shift %9, %9 : (tensor<32x108x103x25xi1>, tensor<32x108x103x25xi1>) -> tensor<32x108x103x25xi1>
    %13 = tosa.abs %11 : (tensor<32x108x103x25xf32>) -> tensor<32x108x103x25xf32>
    %14 = tosa.logical_or %12, %9 : (tensor<32x108x103x25xi1>, tensor<32x108x103x25xi1>) -> tensor<32x108x103x25xi1>
    %15 = tosa.reduce_sum %14 {axis = 3 : i32} : (tensor<32x108x103x25xi1>) -> tensor<32x108x103x1xi1>
    %16 = tosa.pow %13, %13 : (tensor<32x108x103x25xf32>, tensor<32x108x103x25xf32>) -> tensor<32x108x103x25xf32>
    %17 = tosa.argmax %11 {axis = 3 : i32} : (tensor<32x108x103x25xf32>) -> tensor<32x108x103xi32>
    %18 = tosa.maximum %17, %17 : (tensor<32x108x103xi32>, tensor<32x108x103xi32>) -> tensor<32x108x103xi32>
    %t_19 = tosa.const_shape {values = dense<[ 1, 2, 3, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %19 = tosa.tile %16, %t_19 : (tensor<32x108x103x25xf32>, !tosa.shape<4>) -> tensor<32x216x309x75xf32>
    %20 = tosa.clz %17 : (tensor<32x108x103xi32>) -> tensor<32x108x103xi32>
    %21 = tosa.arithmetic_right_shift %20, %18 {round = true} : (tensor<32x108x103xi32>, tensor<32x108x103xi32>) -> tensor<32x108x103xi32>
    return %5, %7, %10, %15, %19, %21 : tensor<i16>, tensor<24x13x84xi1>, tensor<32x108x103x25xf32>, tensor<32x108x103x1xi1>, tensor<32x216x309x75xf32>, tensor<32x108x103xi32>
  }
}
