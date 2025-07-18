module {
  func.func @main(%arg0: tensor<48x1xf32>, %arg1: tensor<65x55xi1>, %arg2: tensor<1x1xi1>) -> (tensor<48xi32>, tensor<1x55xi1>, tensor<1x110xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<48x1xf32>) -> tensor<48x1xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<65x55xi1>, tensor<1x1xi1>) -> tensor<65x55xi1>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<65x55xi1>) -> tensor<65x55xi1>
    %3 = tosa.argmax %0 {axis = 1 : i32} : (tensor<48x1xf32>) -> tensor<48xi32>
    %4 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<65x55xi1>) -> tensor<1x55xi1>
    %5 = tosa.identity %2 : (tensor<65x55xi1>) -> tensor<65x55xi1>
    %6 = tosa.bitwise_not %4 : (tensor<1x55xi1>) -> tensor<1x55xi1>
    %7 = tosa.logical_xor %6, %4 : (tensor<1x55xi1>, tensor<1x55xi1>) -> tensor<1x55xi1>
    %8 = tosa.concat %7, %4 {axis = 1 : i32} : (tensor<1x55xi1>, tensor<1x55xi1>) -> tensor<1x110xi1>
    %9 = tosa.reduce_sum %5 {axis = 0 : i32} : (tensor<65x55xi1>) -> tensor<1x55xi1>
    %10 = tosa.clz %8 : (tensor<1x110xi1>) -> tensor<1x110xi1>
    return %3, %9, %10 : tensor<48xi32>, tensor<1x55xi1>, tensor<1x110xi1>
  }
}
