module {
  func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<7x20x78x76xi1>) -> (tensor<i32>, tensor<7x1x78x76xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = tosa.reduce_any %arg2 {axis = 1 : i32} : (tensor<7x20x78x76xi1>) -> tensor<7x1x78x76xi1>
    %2 = tosa.bitwise_xor %0, %0 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.logical_not %1 : (tensor<7x1x78x76xi1>) -> tensor<7x1x78x76xi1>
    %4 = tosa.add %1, %3 : (tensor<7x1x78x76xi1>, tensor<7x1x78x76xi1>) -> tensor<7x1x78x76xi1>
    %5 = tosa.bitwise_xor %4, %1 : (tensor<7x1x78x76xi1>, tensor<7x1x78x76xi1>) -> tensor<7x1x78x76xi1>
    %6 = tosa.logical_or %5, %4 : (tensor<7x1x78x76xi1>, tensor<7x1x78x76xi1>) -> tensor<7x1x78x76xi1>
    return %2, %6 : tensor<i32>, tensor<7x1x78x76xi1>
  }
}
