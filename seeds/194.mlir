module {
  func.func @main(%arg0: tensor<5x76x29xf32>, %arg1: tensor<5x76x1xf32>) -> (tensor<5x76x29xf32>, tensor<5x76x29xi1>, tensor<5x76x29xf32>, tensor<5x1x29xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<5x76x29xf32>, tensor<5x76x1xf32>) -> tensor<5x76x29xf32>
    %1 = tosa.floor %0 : (tensor<5x76x29xf32>) -> tensor<5x76x29xf32>
    %2 = tosa.maximum %1, %0 : (tensor<5x76x29xf32>, tensor<5x76x29xf32>) -> tensor<5x76x29xf32>
    %3 = tosa.greater %2, %1 : (tensor<5x76x29xf32>, tensor<5x76x29xf32>) -> tensor<5x76x29xi1>
    %4 = tosa.rsqrt %0 : (tensor<5x76x29xf32>) -> tensor<5x76x29xf32>
    %5 = tosa.logical_or %3, %3 : (tensor<5x76x29xi1>, tensor<5x76x29xi1>) -> tensor<5x76x29xi1>
    %6 = tosa.equal %1, %1 : (tensor<5x76x29xf32>, tensor<5x76x29xf32>) -> tensor<5x76x29xi1>
    %7 = tosa.reduce_product %5 {axis = 1 : i32} : (tensor<5x76x29xi1>) -> tensor<5x1x29xi1>
    %8 = tosa.logical_or %7, %7 : (tensor<5x1x29xi1>, tensor<5x1x29xi1>) -> tensor<5x1x29xi1>
    %9 = tosa.exp %0 : (tensor<5x76x29xf32>) -> tensor<5x76x29xf32>
    %10 = tosa.bitwise_or %8, %7 : (tensor<5x1x29xi1>, tensor<5x1x29xi1>) -> tensor<5x1x29xi1>
    return %4, %6, %9, %10 : tensor<5x76x29xf32>, tensor<5x76x29xi1>, tensor<5x76x29xf32>, tensor<5x1x29xi1>
  }
}
