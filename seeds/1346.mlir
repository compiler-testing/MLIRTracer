module {
  func.func @main(%arg0: tensor<19x9x9x59xi1>, %arg1: tensor<1x9x9x59xi1>, %arg2: tensor<94x34x24x8x5xi32>, %arg3: tensor<1x34x1x1x1xi32>) -> (tensor<94x34x24x8x5xi32>, tensor<1x2x2x1xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<19x9x9x59xi1>, tensor<1x9x9x59xi1>) -> tensor<19x9x9x59xi1>
    %1 = tosa.reduce_all %0 {axis = 0 : i32} : (tensor<19x9x9x59xi1>) -> tensor<1x9x9x59xi1>
    %r_2 = tosa.const_shape {values = dense<[ 3, 1593 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %1, %r_2 : (tensor<1x9x9x59xi1>, !tosa.shape<2>) -> tensor<3x1593xi1>
    %3 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %4 = tosa.transpose %2 {perms = array<i32: 0, 1>} : (tensor<3x1593xi1>) -> tensor<3x1593xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<3x1593xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<3x1593xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 0, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_6_size = tosa.const_shape {values = dense<[ 8, 4 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.slice %5, %s_6_start, %s_6_size : (tensor<3x1593xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<8x4xi1>
    %7 = tosa.sub %6, %6 : (tensor<8x4xi1>, tensor<8x4xi1>) -> tensor<8x4xi1>
    %8 = tosa.bitwise_or %7, %6 : (tensor<8x4xi1>, tensor<8x4xi1>) -> tensor<8x4xi1>
    %9 = tosa.sub %8, %6 : (tensor<8x4xi1>, tensor<8x4xi1>) -> tensor<8x4xi1>
    %10 = tosa.logical_or %9, %7 : (tensor<8x4xi1>, tensor<8x4xi1>) -> tensor<8x4xi1>
    %11 = tosa.reduce_min %10 {axis = 0 : i32} : (tensor<8x4xi1>) -> tensor<1x4xi1>
    %12 = tosa.intdiv %arg2, %arg3 : (tensor<94x34x24x8x5xi32>, tensor<1x34x1x1x1xi32>) -> tensor<94x34x24x8x5xi32>
    %r_13 = tosa.const_shape {values = dense<[ 1, 2, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %13 = tosa.reshape %11, %r_13 : (tensor<1x4xi1>, !tosa.shape<4>) -> tensor<1x2x2x1xi1>
    %14 = tosa.clz %13 : (tensor<1x2x2x1xi1>) -> tensor<1x2x2x1xi1>
    return %12, %14 : tensor<94x34x24x8x5xi32>, tensor<1x2x2x1xi1>
  }
}
