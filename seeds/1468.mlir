module {
  func.func @main(%arg0: tensor<9xi1>, %arg1: tensor<64x26x95xf32>) -> (tensor<1xi1>, tensor<1xi1>, tensor<64x26xi32>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<9xi1>) -> tensor<1xi1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.logical_or %1, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %3 = tosa.sub %2, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.sigmoid %arg1 : (tensor<64x26x95xf32>) -> tensor<64x26x95xf32>
    %5 = tosa.reduce_all %3 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.clz %5 : (tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.log %4 : (tensor<64x26x95xf32>) -> tensor<64x26x95xf32>
    %8 = tosa.reduce_max %7 {axis = 2 : i32} : (tensor<64x26x95xf32>) -> tensor<64x26x1xf32>
    %9 = tosa.ceil %8 : (tensor<64x26x1xf32>) -> tensor<64x26x1xf32>
    %10 = tosa.bitwise_or %3, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %11 = tosa.argmax %9 {axis = 2 : i32} : (tensor<64x26x1xf32>) -> tensor<64x26xi32>
    return %6, %10, %11 : tensor<1xi1>, tensor<1xi1>, tensor<64x26xi32>
  }
}
