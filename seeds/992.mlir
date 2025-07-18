module {
  func.func @main(%arg0: tensor<29xf32>, %arg1: tensor<33x29x78x92xi1>, %arg2: tensor<33x29x1x1xi1>) -> (tensor<33x29x78x1xi1>, tensor<33x29x78x92xi1>, tensor<1xf32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<29xf32>) -> tensor<29xf32>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<33x29x78x92xi1>, tensor<33x29x1x1xi1>) -> tensor<33x29x78x92xi1>
    %2 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<29xf32>) -> tensor<1xf32>
    %3 = tosa.clz %1 : (tensor<33x29x78x92xi1>) -> tensor<33x29x78x92xi1>
    %4 = tosa.bitwise_or %1, %3 : (tensor<33x29x78x92xi1>, tensor<33x29x78x92xi1>) -> tensor<33x29x78x92xi1>
    %5 = tosa.logical_xor %4, %3 : (tensor<33x29x78x92xi1>, tensor<33x29x78x92xi1>) -> tensor<33x29x78x92xi1>
    %6 = tosa.reduce_any %5 {axis = 3 : i32} : (tensor<33x29x78x92xi1>) -> tensor<33x29x78x1xi1>
    %7 = tosa.bitwise_xor %3, %1 : (tensor<33x29x78x92xi1>, tensor<33x29x78x92xi1>) -> tensor<33x29x78x92xi1>
    %8 = tosa.minimum %2, %2 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %9 = tosa.floor %8 : (tensor<1xf32>) -> tensor<1xf32>
    return %6, %7, %9 : tensor<33x29x78x1xi1>, tensor<33x29x78x92xi1>, tensor<1xf32>
  }
}
