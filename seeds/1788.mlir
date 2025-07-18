module {
  func.func @main(%arg0: tensor<63x82xi1>, %arg1: tensor<1x82xi1>, %arg2: tensor<61x38x49x57xf32>, %arg3: tensor<95x68x59x17xf32>, %arg4: tensor<95xf32>) -> (tensor<63x82xi1>, tensor<63x82xi1>, tensor<3x1x2140x3xf32>, tensor<11x12x10xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<63x82xi1>, tensor<1x82xi1>) -> tensor<63x82xi1>
    %1 = tosa.exp %arg2 : (tensor<61x38x49x57xf32>) -> tensor<61x38x49x57xf32>
    %2 = tosa.clz %0 : (tensor<63x82xi1>) -> tensor<63x82xi1>
    %input_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.transpose_conv2d %1, %arg3, %arg4, %input_zp_3, %weight_zp_3 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 61, 107, 110, 95>} : (tensor<61x38x49x57xf32>, tensor<95x68x59x17xf32>, tensor<95xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<61x107x110x95xf32>
    %4 = tosa.reverse %1 {axis = 0 : i32} : (tensor<61x38x49x57xf32>) -> tensor<61x38x49x57xf32>
    %in_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5 = tosa.negate %3, %in_zp_5, %out_zp_5 : (tensor<61x107x110x95xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<61x107x110x95xf32>
    %6 = tosa.logical_right_shift %0, %0 : (tensor<63x82xi1>, tensor<63x82xi1>) -> tensor<63x82xi1>
    %7 = tosa.floor %4 : (tensor<61x38x49x57xf32>) -> tensor<61x38x49x57xf32>
    %8 = tosa.logical_right_shift %2, %6 : (tensor<63x82xi1>, tensor<63x82xi1>) -> tensor<63x82xi1>
    %in_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_9 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %9 = tosa.negate %7, %in_zp_9, %out_zp_9 : (tensor<61x38x49x57xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<61x38x49x57xf32>
    %10 = tosa.reduce_min %5 {axis = 3 : i32} : (tensor<61x107x110x95xf32>) -> tensor<61x107x110x1xf32>
    %11 = tosa.concat %10, %10 {axis = 3 : i32} : (tensor<61x107x110x1xf32>, tensor<61x107x110x1xf32>) -> tensor<61x107x110x2xf32>
    %12 = tosa.reverse %11 {axis = 2 : i32} : (tensor<61x107x110x2xf32>) -> tensor<61x107x110x2xf32>
    %13 = tosa.maximum %9, %9 : (tensor<61x38x49x57xf32>, tensor<61x38x49x57xf32>) -> tensor<61x38x49x57xf32>
    %14 = tosa.bitwise_xor %0, %8 : (tensor<63x82xi1>, tensor<63x82xi1>) -> tensor<63x82xi1>
    %r_15 = tosa.const_shape {values = dense<[ 133, 48678, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %15 = tosa.reshape %13, %r_15 : (tensor<61x38x49x57xf32>, !tosa.shape<3>) -> tensor<133x48678x1xf32>
    %16 = tosa.pow %15, %15 : (tensor<133x48678x1xf32>, tensor<133x48678x1xf32>) -> tensor<133x48678x1xf32>
    %17 = tosa.bitwise_not %8 : (tensor<63x82xi1>) -> tensor<63x82xi1>
    %18 = tosa.sub %12, %12 : (tensor<61x107x110x2xf32>, tensor<61x107x110x2xf32>) -> tensor<61x107x110x2xf32>
    %r_19 = tosa.const_shape {values = dense<[ 671, 1, 2140, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %19 = tosa.reshape %18, %r_19 : (tensor<61x107x110x2xf32>, !tosa.shape<4>) -> tensor<671x1x2140x1xf32>
    %20 = tosa.reduce_min %19 {axis = 0 : i32} : (tensor<671x1x2140x1xf32>) -> tensor<1x1x2140x1xf32>
    %t_21 = tosa.const_shape {values = dense<[ 3, 1, 1, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %21 = tosa.tile %20, %t_21 : (tensor<1x1x2140x1xf32>, !tosa.shape<4>) -> tensor<3x1x2140x3xf32>
    %s_22_start = tosa.const_shape {values = dense<[ 43, 115, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_22_size = tosa.const_shape {values = dense<[ 11, 12, 10 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %22 = tosa.slice %16, %s_22_start, %s_22_size : (tensor<133x48678x1xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<11x12x10xf32>
    return %14, %17, %21, %22 : tensor<63x82xi1>, tensor<63x82xi1>, tensor<3x1x2140x3xf32>, tensor<11x12x10xf32>
  }
}
