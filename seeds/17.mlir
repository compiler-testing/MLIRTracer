module {
  func.func @main(%arg0: tensor<94x23x67xi16>, %arg1: tensor<94x67x48xi16>, %arg2: tensor<11x44x96x92xf32>, %arg3: tensor<15x50x19x76xf32>, %arg4: tensor<15xf32>) -> (tensor<94x23x48xi16>, tensor<11x139x118x15xi1>, tensor<11x139x118x15xi1>, tensor<11x139x118x15xf32>, tensor<3x5x1x12xi1>, tensor<902110x3x1xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<94x23x67xi16>, tensor<94x67x48xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<94x23x48xi16>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 2>, stride = array<i64: 2, 1>, out_shape = array<i64: 11, 139, 118, 15>} : (tensor<11x44x96x92xf32>, tensor<15x50x19x76xf32>, tensor<15xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<11x139x118x15xf32>
    %2 = tosa.equal %1, %1 : (tensor<11x139x118x15xf32>, tensor<11x139x118x15xf32>) -> tensor<11x139x118x15xi1>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<11x139x118x15xi1>, tensor<11x139x118x15xi1>) -> tensor<11x139x118x15xi1>
    %4 = tosa.greater %1, %1 : (tensor<11x139x118x15xf32>, tensor<11x139x118x15xf32>) -> tensor<11x139x118x15xi1>
    %5 = tosa.bitwise_not %4 : (tensor<11x139x118x15xi1>) -> tensor<11x139x118x15xi1>
    %6 = tosa.logical_not %5 : (tensor<11x139x118x15xi1>) -> tensor<11x139x118x15xi1>
    %7 = tosa.clamp %1 {min_val = -2.000000e+01 : f32, max_val = 8.900000e+01 : f32} : (tensor<11x139x118x15xf32>) -> tensor<11x139x118x15xf32>
    %8 = tosa.logical_not %6 : (tensor<11x139x118x15xi1>) -> tensor<11x139x118x15xi1>
    %9 = tosa.logical_right_shift %8, %8 : (tensor<11x139x118x15xi1>, tensor<11x139x118x15xi1>) -> tensor<11x139x118x15xi1>
    %10 = tosa.logical_xor %9, %9 : (tensor<11x139x118x15xi1>, tensor<11x139x118x15xi1>) -> tensor<11x139x118x15xi1>
    %11 = tosa.bitwise_or %10, %10 : (tensor<11x139x118x15xi1>, tensor<11x139x118x15xi1>) -> tensor<11x139x118x15xi1>
    %12 = tosa.pow %7, %7 : (tensor<11x139x118x15xf32>, tensor<11x139x118x15xf32>) -> tensor<11x139x118x15xf32>
    %13 = tosa.greater_equal %7, %12 : (tensor<11x139x118x15xf32>, tensor<11x139x118x15xf32>) -> tensor<11x139x118x15xi1>
    %14 = tosa.clz %13 : (tensor<11x139x118x15xi1>) -> tensor<11x139x118x15xi1>
    %15 = tosa.minimum %12, %12 : (tensor<11x139x118x15xf32>, tensor<11x139x118x15xf32>) -> tensor<11x139x118x15xf32>
    %s_16_start = tosa.const_shape {values = dense<[ 8, 8, 4, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_16_size = tosa.const_shape {values = dense<[ 3, 5, 10, 12 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %16 = tosa.slice %13, %s_16_start, %s_16_size : (tensor<11x139x118x15xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x5x10x12xi1>
    %in_zp_17 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_17 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %17 = tosa.negate %14, %in_zp_17, %out_zp_17 : (tensor<11x139x118x15xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<11x139x118x15xi1>
    %18 = tosa.reduce_any %16 {axis = 2 : i32} : (tensor<3x5x10x12xi1>) -> tensor<3x5x1x12xi1>
    %r_19 = tosa.const_shape {values = dense<[ 902110, 3, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %19 = tosa.reshape %17, %r_19 : (tensor<11x139x118x15xi1>, !tosa.shape<3>) -> tensor<902110x3x1xi1>
    return %0, %3, %11, %15, %18, %19 : tensor<94x23x48xi16>, tensor<11x139x118x15xi1>, tensor<11x139x118x15xi1>, tensor<11x139x118x15xf32>, tensor<3x5x1x12xi1>, tensor<902110x3x1xi1>
  }
}
