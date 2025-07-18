module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<1xf32>, %arg2: tensor<95xi1>, %arg3: tensor<95xi1>) -> (tensor<4xi1>, tensor<95xi1>, tensor<1xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
    %1 = tosa.logical_xor %arg2, %arg3 : (tensor<95xi1>, tensor<95xi1>) -> tensor<95xi1>
    %2 = tosa.sub %0, %0 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %3 = tosa.logical_and %1, %1 : (tensor<95xi1>, tensor<95xi1>) -> tensor<95xi1>
    %4 = tosa.abs %1 : (tensor<95xi1>) -> tensor<95xi1>
    %5 = tosa.bitwise_or %3, %1 : (tensor<95xi1>, tensor<95xi1>) -> tensor<95xi1>
    %6 = tosa.equal %0, %2 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    %7 = tosa.arithmetic_right_shift %5, %5 {round = false} : (tensor<95xi1>, tensor<95xi1>) -> tensor<95xi1>
    %8 = tosa.reduce_min %4 {axis = 0 : i32} : (tensor<95xi1>) -> tensor<1xi1>
    %9 = tosa.logical_right_shift %7, %7 : (tensor<95xi1>, tensor<95xi1>) -> tensor<95xi1>
    %10 = tosa.reduce_product %8 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %6, %9, %10 : tensor<4xi1>, tensor<95xi1>, tensor<1xi1>
  }
}
