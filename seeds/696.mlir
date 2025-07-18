module {
  func.func @main(%arg0: tensor<41xi16>, %arg1: tensor<62x73x39xi1>, %arg2: tensor<50xf32>) -> (tensor<i1>, tensor<10x18xi1>, tensor<50xi1>, tensor<50xf32>, tensor<50xf32>, tensor<50xf32>, tensor<50xf32>, tensor<i32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<41xi16>) -> tensor<i32>
    %s_1_start = tosa.const_shape {values = dense<[ 2, 7, 27 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_1_size = tosa.const_shape {values = dense<[ 3, 5, 12 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.slice %arg1, %s_1_start, %s_1_size : (tensor<62x73x39xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<3x5x12xi1>
    %2 = tosa.equal %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<3x5x12xi1>, tensor<3x5x12xi1>) -> tensor<3x5x12xi1>
    %4 = tosa.logical_not %2 : (tensor<i1>) -> tensor<i1>
    %r_5 = tosa.const_shape {values = dense<[ 10, 18 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %3, %r_5 : (tensor<3x5x12xi1>, !tosa.shape<2>) -> tensor<10x18xi1>
    %6 = tosa.floor %arg2 : (tensor<50xf32>) -> tensor<50xf32>
    %7 = tosa.logical_right_shift %5, %5 : (tensor<10x18xi1>, tensor<10x18xi1>) -> tensor<10x18xi1>
    %8 = tosa.log %6 : (tensor<50xf32>) -> tensor<50xf32>
    %9 = tosa.greater %6, %8 : (tensor<50xf32>, tensor<50xf32>) -> tensor<50xi1>
    %10 = tosa.rsqrt %8 : (tensor<50xf32>) -> tensor<50xf32>
    %11 = tosa.maximum %6, %6 : (tensor<50xf32>, tensor<50xf32>) -> tensor<50xf32>
    %12 = tosa.rsqrt %6 : (tensor<50xf32>) -> tensor<50xf32>
    %13 = tosa.reverse %10 {axis = 0 : i32} : (tensor<50xf32>) -> tensor<50xf32>
    %14 = tosa.exp %13 : (tensor<50xf32>) -> tensor<50xf32>
    %15 = tosa.abs %14 : (tensor<50xf32>) -> tensor<50xf32>
    %16 = tosa.pow %12, %10 : (tensor<50xf32>, tensor<50xf32>) -> tensor<50xf32>
    %17 = tosa.minimum %12, %14 : (tensor<50xf32>, tensor<50xf32>) -> tensor<50xf32>
    %18 = tosa.reciprocal %17 : (tensor<50xf32>) -> tensor<50xf32>
    %19 = tosa.intdiv %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %4, %7, %9, %11, %15, %16, %18, %19 : tensor<i1>, tensor<10x18xi1>, tensor<50xi1>, tensor<50xf32>, tensor<50xf32>, tensor<50xf32>, tensor<50xf32>, tensor<i32>
  }
}
