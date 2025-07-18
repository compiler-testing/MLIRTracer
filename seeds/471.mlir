module {
  func.func @main(%arg0: tensor<41x71x50x19x72x37xi32>, %arg1: tensor<41x71x50x19x72x1xi32>, %arg2: tensor<65x48xi1>, %arg3: tensor<65x1xi1>, %arg4: tensor<29x44xf32>) -> (tensor<41x71x50x19x72x37xi32>, tensor<65x1xi1>, tensor<29x44xf32>, tensor<29xi1>, tensor<29x44xf32>, tensor<44xi32>, tensor<1xi32>, tensor<44xi32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<41x71x50x19x72x37xi32>, tensor<41x71x50x19x72x1xi32>) -> tensor<41x71x50x19x72x37xi32>
    %1 = tosa.logical_or %arg2, %arg3 : (tensor<65x48xi1>, tensor<65x1xi1>) -> tensor<65x48xi1>
    %2 = tosa.reduce_all %1 {axis = 1 : i32} : (tensor<65x48xi1>) -> tensor<65x1xi1>
    %3 = tosa.logical_and %2, %2 : (tensor<65x1xi1>, tensor<65x1xi1>) -> tensor<65x1xi1>
    %4 = tosa.clz %3 : (tensor<65x1xi1>) -> tensor<65x1xi1>
    %5 = tosa.rsqrt %arg4 : (tensor<29x44xf32>) -> tensor<29x44xf32>
    %6 = tosa.reverse %5 {axis = 0 : i32} : (tensor<29x44xf32>) -> tensor<29x44xf32>
    %7 = tosa.argmax %5 {axis = 1 : i32} : (tensor<29x44xf32>) -> tensor<29xi32>
    %8 = tosa.equal %7, %7 : (tensor<29xi32>, tensor<29xi32>) -> tensor<29xi1>
    %9 = tosa.argmax %5 {axis = 0 : i32} : (tensor<29x44xf32>) -> tensor<44xi32>
    %10 = tosa.intdiv %9, %9 : (tensor<44xi32>, tensor<44xi32>) -> tensor<44xi32>
    %11 = tosa.reduce_sum %10 {axis = 0 : i32} : (tensor<44xi32>) -> tensor<1xi32>
    %12 = tosa.reverse %9 {axis = 0 : i32} : (tensor<44xi32>) -> tensor<44xi32>
    %13 = tosa.add %9, %12 : (tensor<44xi32>, tensor<44xi32>) -> tensor<44xi32>
    %14 = tosa.arithmetic_right_shift %13, %12 {round = true} : (tensor<44xi32>, tensor<44xi32>) -> tensor<44xi32>
    %15 = tosa.abs %11 : (tensor<1xi32>) -> tensor<1xi32>
    %16 = tosa.floor %5 : (tensor<29x44xf32>) -> tensor<29x44xf32>
    %17 = tosa.bitwise_and %13, %14 : (tensor<44xi32>, tensor<44xi32>) -> tensor<44xi32>
    %18 = tosa.logical_right_shift %17, %17 : (tensor<44xi32>, tensor<44xi32>) -> tensor<44xi32>
    %19 = tosa.logical_left_shift %15, %15 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %20 = tosa.clz %17 : (tensor<44xi32>) -> tensor<44xi32>
    return %0, %4, %6, %8, %16, %18, %19, %20 : tensor<41x71x50x19x72x37xi32>, tensor<65x1xi1>, tensor<29x44xf32>, tensor<29xi1>, tensor<29x44xf32>, tensor<44xi32>, tensor<1xi32>, tensor<44xi32>
  }
}
