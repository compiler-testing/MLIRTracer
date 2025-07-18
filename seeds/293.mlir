module {
  func.func @main(%arg0: tensor<78x15x54x92xf32>, %arg1: tensor<85x64x22x45xf32>, %arg2: tensor<85xf32>, %arg3: tensor<10x36x4xi1>, %arg4: tensor<10x1x1xi1>) -> (tensor<1x95x131x85xf32>, tensor<78x95x131x85xf32>, tensor<3x48xi1>, tensor<1x7x4xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 2, 2, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 78, 95, 131, 85>} : (tensor<78x15x54x92xf32>, tensor<85x64x22x45xf32>, tensor<85xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<78x95x131x85xf32>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<78x95x131x85xf32>) -> tensor<1x95x131x85xf32>
    %2 = tosa.logical_and %arg3, %arg4 : (tensor<10x36x4xi1>, tensor<10x1x1xi1>) -> tensor<10x36x4xi1>
    %3 = tosa.bitwise_not %2 : (tensor<10x36x4xi1>) -> tensor<10x36x4xi1>
    %4 = tosa.add %3, %2 : (tensor<10x36x4xi1>, tensor<10x36x4xi1>) -> tensor<10x36x4xi1>
    %5 = tosa.reduce_any %4 {axis = 0 : i32} : (tensor<10x36x4xi1>) -> tensor<1x36x4xi1>
    %6 = tosa.logical_or %5, %5 : (tensor<1x36x4xi1>, tensor<1x36x4xi1>) -> tensor<1x36x4xi1>
    %7 = tosa.ceil %0 : (tensor<78x95x131x85xf32>) -> tensor<78x95x131x85xf32>
    %8 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<10x36x4xi1>) -> tensor<1x36x4xi1>
    %9 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %10 = tosa.transpose %8 {perms = array<i32: 0, 2, 1>} : (tensor<1x36x4xi1>) -> tensor<1x4x36xi1>
    %r_11 = tosa.const_shape {values = dense<[ 3, 48 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %11 = tosa.reshape %6, %r_11 : (tensor<1x36x4xi1>, !tosa.shape<2>) -> tensor<3x48xi1>
    %12 = tosa.reverse %10 {axis = 0 : i32} : (tensor<1x4x36xi1>) -> tensor<1x4x36xi1>
    %s_13_start = tosa.const_shape {values = dense<[ 0, 0, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_13_size = tosa.const_shape {values = dense<[ 1, 7, 4 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %13 = tosa.slice %12, %s_13_start, %s_13_size : (tensor<1x4x36xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x7x4xi1>
    %in_zp_14 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_14 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %14 = tosa.negate %13, %in_zp_14, %out_zp_14 : (tensor<1x7x4xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1x7x4xi1>
    return %1, %7, %11, %14 : tensor<1x95x131x85xf32>, tensor<78x95x131x85xf32>, tensor<3x48xi1>, tensor<1x7x4xi1>
  }
}
