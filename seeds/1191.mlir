module {
  func.func @main(%arg0: tensor<55xf32>, %arg1: tensor<i64>) -> (tensor<i64>, tensor<55xf32>, tensor<55xf32>, tensor<1xi1>, tensor<55xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<55xf32>) -> tensor<55xf32>
    %1 = tosa.reciprocal %0 : (tensor<55xf32>) -> tensor<55xf32>
    %2 = tosa.clz %arg1 : (tensor<i64>) -> tensor<i64>
    %3 = tosa.greater %1, %0 : (tensor<55xf32>, tensor<55xf32>) -> tensor<55xi1>
    %4 = tosa.floor %1 : (tensor<55xf32>) -> tensor<55xf32>
    %5 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<55xi1>) -> tensor<1xi1>
    %6 = tosa.bitwise_xor %5, %5 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.maximum %1, %1 : (tensor<55xf32>, tensor<55xf32>) -> tensor<55xf32>
    %8 = tosa.sigmoid %1 : (tensor<55xf32>) -> tensor<55xf32>
    %9 = tosa.sub %6, %6 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %10 = tosa.logical_or %9, %9 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %11 = tosa.ceil %8 : (tensor<55xf32>) -> tensor<55xf32>
    return %2, %4, %7, %10, %11 : tensor<i64>, tensor<55xf32>, tensor<55xf32>, tensor<1xi1>, tensor<55xf32>
  }
}
