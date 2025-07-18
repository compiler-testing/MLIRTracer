module {
  func.func @main(%arg0: tensor<80x44xf32>, %arg1: tensor<3xi1>, %arg2: tensor<3xi1>) -> (tensor<80x1xf32>, tensor<3xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<80x44xf32>) -> tensor<80x44xf32>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<80x44xf32>) -> tensor<80x1xf32>
    %2 = tosa.logical_xor %arg1, %arg2 : (tensor<3xi1>, tensor<3xi1>) -> tensor<3xi1>
    %3 = tosa.bitwise_not %2 : (tensor<3xi1>) -> tensor<3xi1>
    return %1, %3 : tensor<80x1xf32>, tensor<3xi1>
  }
}
