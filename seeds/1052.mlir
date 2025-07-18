module {
  func.func @main(%arg0: tensor<82xf32>, %arg1: tensor<82xf32>, %arg2: tensor<39x69xi8>, %arg3: tensor<39x69xi8>, %arg4: tensor<9x41xi1>, %arg5: tensor<9x1xi1>) -> (tensor<39x69xi1>, tensor<1x1xf32>, tensor<1x41xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<82xf32>, tensor<82xf32>) -> tensor<82xf32>
    %1 = tosa.bitwise_and %arg2, %arg3 : (tensor<39x69xi8>, tensor<39x69xi8>) -> tensor<39x69xi8>
    %2 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<82xf32>) -> tensor<1xf32>
    %3 = tosa.reciprocal %2 : (tensor<1xf32>) -> tensor<1xf32>
    %4 = tosa.rsqrt %3 : (tensor<1xf32>) -> tensor<1xf32>
    %r_5 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %4, %r_5 : (tensor<1xf32>, !tosa.shape<2>) -> tensor<1x1xf32>
    %6 = tosa.logical_xor %arg4, %arg5 : (tensor<9x41xi1>, tensor<9x1xi1>) -> tensor<9x41xi1>
    %7 = tosa.ceil %5 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %8 = tosa.greater_equal %1, %1 : (tensor<39x69xi8>, tensor<39x69xi8>) -> tensor<39x69xi1>
    %9 = tosa.bitwise_or %8, %8 : (tensor<39x69xi1>, tensor<39x69xi1>) -> tensor<39x69xi1>
    %10 = tosa.bitwise_not %9 : (tensor<39x69xi1>) -> tensor<39x69xi1>
    %11 = tosa.sigmoid %7 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %12 = tosa.arithmetic_right_shift %6, %6 {round = false} : (tensor<9x41xi1>, tensor<9x41xi1>) -> tensor<9x41xi1>
    %13 = tosa.rsqrt %11 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %14 = tosa.reduce_any %12 {axis = 0 : i32} : (tensor<9x41xi1>) -> tensor<1x41xi1>
    return %10, %13, %14 : tensor<39x69xi1>, tensor<1x1xf32>, tensor<1x41xi1>
  }
}
