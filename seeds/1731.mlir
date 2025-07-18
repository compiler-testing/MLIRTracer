module {
  func.func @main(%arg0: tensor<45x30xi1>, %arg1: tensor<45x30xi1>, %arg2: tensor<65xi32>, %arg3: tensor<65xi32>) -> (tensor<1x1xi1>, tensor<1xi32>, tensor<1xi32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<45x30xi1>, tensor<45x30xi1>) -> tensor<45x30xi1>
    %1 = tosa.minimum %arg2, %arg3 : (tensor<65xi32>, tensor<65xi32>) -> tensor<65xi32>
    %2 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<65xi32>) -> tensor<1xi32>
    %3 = tosa.reduce_all %0 {axis = 0 : i32} : (tensor<45x30xi1>) -> tensor<1x30xi1>
    %4 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<1x30xi1>) -> tensor<1x1xi1>
    %5 = tosa.bitwise_xor %4, %4 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %6 = tosa.logical_or %5, %5 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %7 = tosa.reduce_max %6 {axis = 0 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %8 = tosa.logical_not %7 : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %9 = tosa.logical_left_shift %2, %2 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %10 = tosa.maximum %9, %2 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %11 = tosa.bitwise_and %9, %2 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    return %8, %10, %11 : tensor<1x1xi1>, tensor<1xi32>, tensor<1xi32>
  }
}
