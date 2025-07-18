module {
  func.func @main(%arg0: tensor<65x93xi32>, %arg1: tensor<1x93xi32>, %arg2: tensor<89x15x93xf32>) -> (tensor<65x93xi1>, tensor<65x1xi1>, tensor<15x93xi32>, tensor<1x15x93xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<65x93xi32>, tensor<1x93xi32>) -> tensor<65x93xi1>
    %1 = tosa.ceil %arg2 : (tensor<89x15x93xf32>) -> tensor<89x15x93xf32>
    %2 = tosa.bitwise_or %0, %0 : (tensor<65x93xi1>, tensor<65x93xi1>) -> tensor<65x93xi1>
    %3 = tosa.pow %1, %1 : (tensor<89x15x93xf32>, tensor<89x15x93xf32>) -> tensor<89x15x93xf32>
    %4 = tosa.ceil %3 : (tensor<89x15x93xf32>) -> tensor<89x15x93xf32>
    %5 = tosa.concat %1, %4 {axis = 0 : i32} : (tensor<89x15x93xf32>, tensor<89x15x93xf32>) -> tensor<178x15x93xf32>
    %6 = tosa.arithmetic_right_shift %2, %0 {round = true} : (tensor<65x93xi1>, tensor<65x93xi1>) -> tensor<65x93xi1>
    %7 = tosa.minimum %5, %5 : (tensor<178x15x93xf32>, tensor<178x15x93xf32>) -> tensor<178x15x93xf32>
    %8 = tosa.reduce_any %0 {axis = 1 : i32} : (tensor<65x93xi1>) -> tensor<65x1xi1>
    %9 = tosa.argmax %7 {axis = 0 : i32} : (tensor<178x15x93xf32>) -> tensor<15x93xi32>
    %10 = tosa.rsqrt %4 : (tensor<89x15x93xf32>) -> tensor<89x15x93xf32>
    %11 = tosa.reduce_max %10 {axis = 0 : i32} : (tensor<89x15x93xf32>) -> tensor<1x15x93xf32>
    return %6, %8, %9, %11 : tensor<65x93xi1>, tensor<65x1xi1>, tensor<15x93xi32>, tensor<1x15x93xf32>
  }
}
