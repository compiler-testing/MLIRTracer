module {
  func.func @main(%arg0: tensor<31x66xi16>, %arg1: tensor<87x8x92xf32>, %arg2: tensor<i1>, %arg3: tensor<i1>) -> (tensor<66xi32>, tensor<87x8x92xf32>, tensor<i1>, tensor<87x1x92xi1>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<31x66xi16>) -> tensor<66xi32>
    %1 = tosa.tanh %arg1 : (tensor<87x8x92xf32>) -> tensor<87x8x92xf32>
    %2 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<66xi32>, tensor<66xi32>) -> tensor<66xi32>
    %3 = tosa.exp %1 : (tensor<87x8x92xf32>) -> tensor<87x8x92xf32>
    %4 = tosa.abs %1 : (tensor<87x8x92xf32>) -> tensor<87x8x92xf32>
    %5 = tosa.logical_and %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %6 = tosa.reduce_max %4 {axis = 1 : i32} : (tensor<87x8x92xf32>) -> tensor<87x1x92xf32>
    %7 = tosa.greater %6, %6 : (tensor<87x1x92xf32>, tensor<87x1x92xf32>) -> tensor<87x1x92xi1>
    %8 = tosa.logical_not %5 : (tensor<i1>) -> tensor<i1>
    %9 = tosa.add %7, %7 : (tensor<87x1x92xi1>, tensor<87x1x92xi1>) -> tensor<87x1x92xi1>
    %10 = tosa.logical_left_shift %8, %8 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %11 = tosa.add %9, %7 : (tensor<87x1x92xi1>, tensor<87x1x92xi1>) -> tensor<87x1x92xi1>
    return %2, %3, %10, %11 : tensor<66xi32>, tensor<87x8x92xf32>, tensor<i1>, tensor<87x1x92xi1>
  }
}
