module {
  func.func @main(%arg0: tensor<98x68xi32>, %arg1: tensor<98x75xi32>, %arg2: tensor<65x26x3xf32>, %arg3: tensor<53x39xi1>, %arg4: tensor<1x39xi1>) -> (tensor<98x143xi32>, tensor<53x39xi1>, tensor<65x26x3xi1>, tensor<65x26x3xf32>, tensor<53x39xi1>, tensor<53x39xi1>, tensor<65x26x3xf32>, tensor<65x26x3xf32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<98x68xi32>, tensor<98x75xi32>) -> tensor<98x143xi32>
    %1 = tosa.abs %0 : (tensor<98x143xi32>) -> tensor<98x143xi32>
    %2 = tosa.ceil %arg2 : (tensor<65x26x3xf32>) -> tensor<65x26x3xf32>
    %3 = tosa.logical_and %arg3, %arg4 : (tensor<53x39xi1>, tensor<1x39xi1>) -> tensor<53x39xi1>
    %4 = tosa.logical_not %3 : (tensor<53x39xi1>) -> tensor<53x39xi1>
    %5 = tosa.greater %2, %2 : (tensor<65x26x3xf32>, tensor<65x26x3xf32>) -> tensor<65x26x3xi1>
    %6 = tosa.log %2 : (tensor<65x26x3xf32>) -> tensor<65x26x3xf32>
    %7 = tosa.arithmetic_right_shift %3, %3 {round = true} : (tensor<53x39xi1>, tensor<53x39xi1>) -> tensor<53x39xi1>
    %8 = tosa.log %2 : (tensor<65x26x3xf32>) -> tensor<65x26x3xf32>
    %9 = tosa.bitwise_or %3, %7 : (tensor<53x39xi1>, tensor<53x39xi1>) -> tensor<53x39xi1>
    %10 = tosa.clz %3 : (tensor<53x39xi1>) -> tensor<53x39xi1>
    %11 = tosa.rsqrt %2 : (tensor<65x26x3xf32>) -> tensor<65x26x3xf32>
    %12 = tosa.pow %2, %6 : (tensor<65x26x3xf32>, tensor<65x26x3xf32>) -> tensor<65x26x3xf32>
    %13 = tosa.floor %11 : (tensor<65x26x3xf32>) -> tensor<65x26x3xf32>
    return %1, %4, %5, %8, %9, %10, %12, %13 : tensor<98x143xi32>, tensor<53x39xi1>, tensor<65x26x3xi1>, tensor<65x26x3xf32>, tensor<53x39xi1>, tensor<53x39xi1>, tensor<65x26x3xf32>, tensor<65x26x3xf32>
  }
}
