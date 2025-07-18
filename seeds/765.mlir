module {
  func.func @main(%arg0: tensor<54xi32>, %arg1: tensor<54xi32>, %arg2: tensor<87x97x36x3x22xf32>) -> (tensor<108xi1>, tensor<87x97x36x3x22xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<54xi32>, tensor<54xi32>) -> tensor<54xi1>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<54xi1>) -> tensor<54xi1>
    %2 = tosa.bitwise_or %1, %0 : (tensor<54xi1>, tensor<54xi1>) -> tensor<54xi1>
    %3 = tosa.logical_right_shift %2, %1 : (tensor<54xi1>, tensor<54xi1>) -> tensor<54xi1>
    %4 = tosa.concat %3, %1 {axis = 0 : i32} : (tensor<54xi1>, tensor<54xi1>) -> tensor<108xi1>
    %5 = tosa.ceil %arg2 : (tensor<87x97x36x3x22xf32>) -> tensor<87x97x36x3x22xf32>
    return %4, %5 : tensor<108xi1>, tensor<87x97x36x3x22xf32>
  }
}
