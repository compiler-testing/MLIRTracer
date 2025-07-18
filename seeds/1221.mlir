module {
  func.func @main(%arg0: tensor<16x35x66xf32>, %arg1: tensor<88x45x95xi1>, %arg2: tensor<88x45x95xi1>) -> (tensor<245x2x1xf32>, tensor<7x10x7xi1>, tensor<88x45xi32>, tensor<7x7x10xi1>, tensor<1x45xi32>, tensor<1x1xi1>, tensor<1xi32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 9, 12, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_0_size = tosa.const_shape {values = dense<[ 7, 7, 10 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<16x35x66xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<7x7x10xf32>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<7x7x10xf32>) -> tensor<7x7x10xf32>
    %2 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 0, 2, 1>} : (tensor<7x7x10xf32>) -> tensor<7x10x7xf32>
    %r_4 = tosa.const_shape {values = dense<[ 245, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.reshape %3, %r_4 : (tensor<7x10x7xf32>, !tosa.shape<3>) -> tensor<245x2x1xf32>
    %5 = tosa.logical_and %arg1, %arg2 : (tensor<88x45x95xi1>, tensor<88x45x95xi1>) -> tensor<88x45x95xi1>
    %6 = tosa.bitwise_and %5, %5 : (tensor<88x45x95xi1>, tensor<88x45x95xi1>) -> tensor<88x45x95xi1>
    %7 = tosa.greater %3, %3 : (tensor<7x10x7xf32>, tensor<7x10x7xf32>) -> tensor<7x10x7xi1>
    %8 = tosa.clz %6 : (tensor<88x45x95xi1>) -> tensor<88x45x95xi1>
    %9 = tosa.bitwise_and %5, %5 : (tensor<88x45x95xi1>, tensor<88x45x95xi1>) -> tensor<88x45x95xi1>
    %10 = tosa.bitwise_xor %8, %5 : (tensor<88x45x95xi1>, tensor<88x45x95xi1>) -> tensor<88x45x95xi1>
    %11 = tosa.reduce_all %9 {axis = 2 : i32} : (tensor<88x45x95xi1>) -> tensor<88x45x1xi1>
    %12 = tosa.argmax %10 {axis = 2 : i32} : (tensor<88x45x95xi1>) -> tensor<88x45xi32>
    %13 = tosa.logical_not %11 : (tensor<88x45x1xi1>) -> tensor<88x45x1xi1>
    %14 = tosa.equal %0, %1 : (tensor<7x7x10xf32>, tensor<7x7x10xf32>) -> tensor<7x7x10xi1>
    %r_15 = tosa.const_shape {values = dense<[ 11, 360 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %15 = tosa.reshape %11, %r_15 : (tensor<88x45x1xi1>, !tosa.shape<2>) -> tensor<11x360xi1>
    %16 = tosa.reduce_min %13 {axis = 0 : i32} : (tensor<88x45x1xi1>) -> tensor<1x45x1xi1>
    %17 = tosa.reduce_all %15 {axis = 0 : i32} : (tensor<11x360xi1>) -> tensor<1x360xi1>
    %18 = tosa.argmax %16 {axis = 2 : i32} : (tensor<1x45x1xi1>) -> tensor<1x45xi32>
    %in_zp_19 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_19 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %19 = tosa.negate %18, %in_zp_19, %out_zp_19 : (tensor<1x45xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x45xi32>
    %20 = tosa.logical_right_shift %19, %19 : (tensor<1x45xi32>, tensor<1x45xi32>) -> tensor<1x45xi32>
    %21 = tosa.logical_and %17, %17 : (tensor<1x360xi1>, tensor<1x360xi1>) -> tensor<1x360xi1>
    %22 = tosa.reduce_product %17 {axis = 1 : i32} : (tensor<1x360xi1>) -> tensor<1x1xi1>
    %23 = tosa.argmax %21 {axis = 1 : i32} : (tensor<1x360xi1>) -> tensor<1xi32>
    return %4, %7, %12, %14, %20, %22, %23 : tensor<245x2x1xf32>, tensor<7x10x7xi1>, tensor<88x45xi32>, tensor<7x7x10xi1>, tensor<1x45xi32>, tensor<1x1xi1>, tensor<1xi32>
  }
}
