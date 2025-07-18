module {
  func.func @main(%arg0: tensor<79x25xf32>, %arg1: tensor<64x21x39xi1>, %arg2: tensor<31x90x25x29x17x57xi32>, %arg3: tensor<1x90x1x29x1x57xi32>) -> (tensor<1x25xf32>, tensor<64x1x39xi1>, tensor<1x25xf32>, tensor<1x25xf32>, tensor<11x1xi1>, tensor<31x90x25x29x17x57xi32>, tensor<1xi32>, tensor<64x1x1xi1>, tensor<1x25xf32>, tensor<31x90x25x29x17x57xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<79x25xf32>) -> tensor<1x25xf32>
    %1 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<64x21x39xi1>) -> tensor<64x1x39xi1>
    %2 = tosa.identity %0 : (tensor<1x25xf32>) -> tensor<1x25xf32>
    %3 = tosa.reciprocal %2 : (tensor<1x25xf32>) -> tensor<1x25xf32>
    %4 = tosa.greater %3, %0 : (tensor<1x25xf32>, tensor<1x25xf32>) -> tensor<1x25xi1>
    %5 = tosa.logical_or %1, %1 : (tensor<64x1x39xi1>, tensor<64x1x39xi1>) -> tensor<64x1x39xi1>
    %6 = tosa.ceil %2 : (tensor<1x25xf32>) -> tensor<1x25xf32>
    %7 = tosa.logical_xor %5, %5 : (tensor<64x1x39xi1>, tensor<64x1x39xi1>) -> tensor<64x1x39xi1>
    %8 = tosa.sub %4, %4 : (tensor<1x25xi1>, tensor<1x25xi1>) -> tensor<1x25xi1>
    %9 = tosa.bitwise_or %7, %5 : (tensor<64x1x39xi1>, tensor<64x1x39xi1>) -> tensor<64x1x39xi1>
    %10 = tosa.bitwise_xor %8, %8 : (tensor<1x25xi1>, tensor<1x25xi1>) -> tensor<1x25xi1>
    %11 = tosa.reciprocal %3 : (tensor<1x25xf32>) -> tensor<1x25xf32>
    %12 = tosa.clz %9 : (tensor<64x1x39xi1>) -> tensor<64x1x39xi1>
    %s_13_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_13_size = tosa.const_shape {values = dense<[ 11, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %13 = tosa.slice %10, %s_13_start, %s_13_size : (tensor<1x25xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<11x2xi1>
    %14 = tosa.logical_or %13, %13 : (tensor<11x2xi1>, tensor<11x2xi1>) -> tensor<11x2xi1>
    %15 = tosa.logical_not %14 : (tensor<11x2xi1>) -> tensor<11x2xi1>
    %16 = tosa.ceil %0 : (tensor<1x25xf32>) -> tensor<1x25xf32>
    %17 = tosa.tanh %2 : (tensor<1x25xf32>) -> tensor<1x25xf32>
    %18 = tosa.intdiv %arg2, %arg3 : (tensor<31x90x25x29x17x57xi32>, tensor<1x90x1x29x1x57xi32>) -> tensor<31x90x25x29x17x57xi32>
    %19 = tosa.reduce_product %15 {axis = 1 : i32} : (tensor<11x2xi1>) -> tensor<11x1xi1>
    %20 = tosa.arithmetic_right_shift %18, %18 {round = true} : (tensor<31x90x25x29x17x57xi32>, tensor<31x90x25x29x17x57xi32>) -> tensor<31x90x25x29x17x57xi32>
    %21 = tosa.greater_equal %18, %18 : (tensor<31x90x25x29x17x57xi32>, tensor<31x90x25x29x17x57xi32>) -> tensor<31x90x25x29x17x57xi1>
    %22 = tosa.argmax %10 {axis = 1 : i32} : (tensor<1x25xi1>) -> tensor<1xi32>
    %23 = tosa.reduce_product %5 {axis = 2 : i32} : (tensor<64x1x39xi1>) -> tensor<64x1x1xi1>
    %24 = tosa.exp %6 : (tensor<1x25xf32>) -> tensor<1x25xf32>
    %25 = tosa.add %21, %21 : (tensor<31x90x25x29x17x57xi1>, tensor<31x90x25x29x17x57xi1>) -> tensor<31x90x25x29x17x57xi1>
    return %11, %12, %16, %17, %19, %20, %22, %23, %24, %25 : tensor<1x25xf32>, tensor<64x1x39xi1>, tensor<1x25xf32>, tensor<1x25xf32>, tensor<11x1xi1>, tensor<31x90x25x29x17x57xi32>, tensor<1xi32>, tensor<64x1x1xi1>, tensor<1x25xf32>, tensor<31x90x25x29x17x57xi1>
  }
}
