module {
  func.func @main(%arg0: tensor<10xi64>, %arg1: tensor<83x37x23x44xf32>, %arg2: tensor<63x18x4xi1>, %arg3: tensor<63x1x4xi1>) -> (tensor<83x37x23x44xf32>, tensor<1xi64>, tensor<63x18x4xi1>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<10xi64>) -> tensor<1xi64>
    %1 = tosa.floor %arg1 : (tensor<83x37x23x44xf32>) -> tensor<83x37x23x44xf32>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<63x18x4xi1>, tensor<63x1x4xi1>) -> tensor<63x18x4xi1>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<63x18x4xi1>, tensor<63x18x4xi1>) -> tensor<63x18x4xi1>
    %4 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<1xi64>) -> tensor<1xi64>
    %5 = tosa.arithmetic_right_shift %4, %0 {round = true} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %6 = tosa.bitwise_not %3 : (tensor<63x18x4xi1>) -> tensor<63x18x4xi1>
    return %1, %5, %6 : tensor<83x37x23x44xf32>, tensor<1xi64>, tensor<63x18x4xi1>
  }
}
