module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<36xf32>) -> (tensor<i32>, tensor<i1>, tensor<i1>, tensor<1xf32>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = tosa.greater %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = tosa.greater %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = tosa.sub %1, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.greater_equal %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = tosa.add %3, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %6 = tosa.bitwise_xor %5, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.arithmetic_right_shift %6, %5 {round = false} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %8 = tosa.bitwise_not %7 : (tensor<i1>) -> tensor<i1>
    %9 = tosa.intdiv %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %10 = tosa.reciprocal %arg2 : (tensor<36xf32>) -> tensor<36xf32>
    %11 = tosa.logical_left_shift %8, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %r_12 = tosa.const_shape {values = dense<[ 36 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %12 = tosa.reshape %10, %r_12 : (tensor<36xf32>, !tosa.shape<1>) -> tensor<36xf32>
    %13 = tosa.bitwise_and %11, %5 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %14 = tosa.add %13, %6 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %15 = tosa.bitwise_xor %14, %5 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %16 = tosa.logical_not %15 : (tensor<i1>) -> tensor<i1>
    %in_zp_17 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_17 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %17 = tosa.negate %12, %in_zp_17, %out_zp_17 : (tensor<36xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<36xf32>
    %18 = tosa.ceil %17 : (tensor<36xf32>) -> tensor<36xf32>
    %19 = tosa.logical_or %3, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %20 = tosa.maximum %18, %12 : (tensor<36xf32>, tensor<36xf32>) -> tensor<36xf32>
    %s_21_start = tosa.const_shape {values = dense<[ 19 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_21_size = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %21 = tosa.slice %20, %s_21_start, %s_21_size : (tensor<36xf32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<1xf32>
    %22 = tosa.sigmoid %21 : (tensor<1xf32>) -> tensor<1xf32>
    return %9, %16, %19, %22 : tensor<i32>, tensor<i1>, tensor<i1>, tensor<1xf32>
  }
}
