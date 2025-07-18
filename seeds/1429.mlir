module {
  func.func @main(%arg0: tensor<2x11x44xi8>, %arg1: tensor<2x11x44xi8>, %arg2: tensor<91x46x66x40x67xf32>, %arg3: tensor<24x22x18xi32>, %arg4: tensor<24x1x1xi32>) -> (tensor<91x46x66x40x67xf32>, tensor<1x11x44xi8>, tensor<24x22x18xi32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<2x11x44xi8>, tensor<2x11x44xi8>) -> tensor<2x11x44xi8>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<2x11x44xi8>) -> tensor<1x11x44xi8>
    %2 = tosa.add %1, %1 : (tensor<1x11x44xi8>, tensor<1x11x44xi8>) -> tensor<1x11x44xi8>
    %3 = tosa.identity %2 : (tensor<1x11x44xi8>) -> tensor<1x11x44xi8>
    %4 = tosa.exp %arg2 : (tensor<91x46x66x40x67xf32>) -> tensor<91x46x66x40x67xf32>
    %5 = tosa.arithmetic_right_shift %3, %1 {round = true} : (tensor<1x11x44xi8>, tensor<1x11x44xi8>) -> tensor<1x11x44xi8>
    %6 = tosa.intdiv %arg3, %arg4 : (tensor<24x22x18xi32>, tensor<24x1x1xi32>) -> tensor<24x22x18xi32>
    return %4, %5, %6 : tensor<91x46x66x40x67xf32>, tensor<1x11x44xi8>, tensor<24x22x18xi32>
  }
}
