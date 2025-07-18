module {
  func.func @main(%arg0: tensor<31xf32>, %arg1: tensor<97x17x26x2xi1>) -> (tensor<3x1xi32>, tensor<31xf32>, tensor<97x17x26x1xi1>, tensor<31xf32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<31xf32>) -> tensor<31xf32>
    %1 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0>} : (tensor<31xf32>) -> tensor<31xf32>
    %3 = tosa.reduce_all %arg1 {axis = 3 : i32} : (tensor<97x17x26x2xi1>) -> tensor<97x17x26x1xi1>
    %4 = tosa.argmax %2 {axis = 0 : i32} : (tensor<31xf32>) -> tensor<i32>
    %r_5 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %4, %r_5 : (tensor<i32>, !tosa.shape<2>) -> tensor<1x1xi32>
    %t_6 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.tile %5, %t_6 : (tensor<1x1xi32>, !tosa.shape<2>) -> tensor<3x1xi32>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %7 = tosa.negate %3, %in_zp_7, %out_zp_7 : (tensor<97x17x26x1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<97x17x26x1xi1>
    %8 = tosa.pow %0, %2 : (tensor<31xf32>, tensor<31xf32>) -> tensor<31xf32>
    %9 = tosa.logical_or %7, %7 : (tensor<97x17x26x1xi1>, tensor<97x17x26x1xi1>) -> tensor<97x17x26x1xi1>
    %10 = tosa.bitwise_xor %9, %3 : (tensor<97x17x26x1xi1>, tensor<97x17x26x1xi1>) -> tensor<97x17x26x1xi1>
    %11 = tosa.exp %0 : (tensor<31xf32>) -> tensor<31xf32>
    return %6, %8, %10, %11 : tensor<3x1xi32>, tensor<31xf32>, tensor<97x17x26x1xi1>, tensor<31xf32>
  }
}
