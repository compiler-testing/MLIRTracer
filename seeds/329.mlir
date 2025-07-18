module {
  func.func @main(%arg0: tensor<100xi1>, %arg1: tensor<18x97x53x53x92xf32>) -> (tensor<18x97x53x53x92xf32>, tensor<1xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<100xi1>) -> tensor<1xi1>
    %1 = tosa.rsqrt %arg1 : (tensor<18x97x53x53x92xf32>) -> tensor<18x97x53x53x92xf32>
    %2 = tosa.logical_left_shift %0, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %1, %2 : tensor<18x97x53x53x92xf32>, tensor<1xi1>
  }
}
