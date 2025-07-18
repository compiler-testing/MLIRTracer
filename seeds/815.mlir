module {
  func.func @main(%arg0: tensor<84x23x19x54x21xi64>, %arg1: tensor<95xf32>, %arg2: tensor<i1>, %arg3: tensor<i1>, %arg4: tensor<53x64x17x26xi32>, %arg5: tensor<53x64x17x26xi32>) -> (tensor<84x23x19x54x21xi64>, tensor<i1>, tensor<95xi1>, tensor<i1>, tensor<159x128x17x78xi32>, tensor<95xf32>, tensor<53x64x17x52xi32>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<84x23x19x54x21xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<84x23x19x54x21xi64>
    %1 = tosa.tanh %arg1 : (tensor<95xf32>) -> tensor<95xf32>
    %2 = tosa.bitwise_and %0, %0 : (tensor<84x23x19x54x21xi64>, tensor<84x23x19x54x21xi64>) -> tensor<84x23x19x54x21xi64>
    %3 = tosa.rsqrt %1 : (tensor<95xf32>) -> tensor<95xf32>
    %t_4 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.tile %3, %t_4 : (tensor<95xf32>, !tosa.shape<1>) -> tensor<95xf32>
    %5 = tosa.abs %4 : (tensor<95xf32>) -> tensor<95xf32>
    %6 = tosa.logical_and %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.bitwise_and %6, %6 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %8 = tosa.equal %1, %5 : (tensor<95xf32>, tensor<95xf32>) -> tensor<95xi1>
    %9 = tosa.arithmetic_right_shift %6, %6 {round = true} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %10 = tosa.pow %1, %4 : (tensor<95xf32>, tensor<95xf32>) -> tensor<95xf32>
    %11 = tosa.intdiv %arg4, %arg5 : (tensor<53x64x17x26xi32>, tensor<53x64x17x26xi32>) -> tensor<53x64x17x26xi32>
    %t_12 = tosa.const_shape {values = dense<[ 3, 2, 1, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %12 = tosa.tile %11, %t_12 : (tensor<53x64x17x26xi32>, !tosa.shape<4>) -> tensor<159x128x17x78xi32>
    %13 = tosa.reciprocal %10 : (tensor<95xf32>) -> tensor<95xf32>
    %14 = tosa.concat %11, %11 {axis = 3 : i32} : (tensor<53x64x17x26xi32>, tensor<53x64x17x26xi32>) -> tensor<53x64x17x52xi32>
    return %2, %7, %8, %9, %12, %13, %14 : tensor<84x23x19x54x21xi64>, tensor<i1>, tensor<95xi1>, tensor<i1>, tensor<159x128x17x78xi32>, tensor<95xf32>, tensor<53x64x17x52xi32>
  }
}
