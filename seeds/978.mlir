module {
  func.func @main(%arg0: tensor<25x57x22xf32>, %arg1: tensor<30x16x23x82xi1>) -> (tensor<1x16x23xi32>, tensor<25x57x1xf32>, tensor<30x16x23x82xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 2 : i32} : (tensor<25x57x22xf32>) -> tensor<25x57x1xf32>
    %1 = tosa.log %0 : (tensor<25x57x1xf32>) -> tensor<25x57x1xf32>
    %2 = tosa.logical_not %arg1 : (tensor<30x16x23x82xi1>) -> tensor<30x16x23x82xi1>
    %3 = tosa.minimum %1, %0 : (tensor<25x57x1xf32>, tensor<25x57x1xf32>) -> tensor<25x57x1xf32>
    %4 = tosa.exp %3 : (tensor<25x57x1xf32>) -> tensor<25x57x1xf32>
    %5 = tosa.logical_left_shift %2, %2 : (tensor<30x16x23x82xi1>, tensor<30x16x23x82xi1>) -> tensor<30x16x23x82xi1>
    %6 = tosa.logical_or %2, %2 : (tensor<30x16x23x82xi1>, tensor<30x16x23x82xi1>) -> tensor<30x16x23x82xi1>
    %7 = tosa.logical_left_shift %6, %2 : (tensor<30x16x23x82xi1>, tensor<30x16x23x82xi1>) -> tensor<30x16x23x82xi1>
    %8 = tosa.argmax %7 {axis = 3 : i32} : (tensor<30x16x23x82xi1>) -> tensor<30x16x23xi32>
    %9 = tosa.reduce_max %8 {axis = 0 : i32} : (tensor<30x16x23xi32>) -> tensor<1x16x23xi32>
    %10 = tosa.tanh %4 : (tensor<25x57x1xf32>) -> tensor<25x57x1xf32>
    %11 = tosa.logical_xor %5, %6 : (tensor<30x16x23x82xi1>, tensor<30x16x23x82xi1>) -> tensor<30x16x23x82xi1>
    return %9, %10, %11 : tensor<1x16x23xi32>, tensor<25x57x1xf32>, tensor<30x16x23x82xi1>
  }
}
