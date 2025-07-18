module {
  func.func @main(%arg0: tensor<96x61x97x43xi1>, %arg1: tensor<43x7x27xf32>, %arg2: tensor<1x1x27xf32>) -> (tensor<96x1x1x43xi1>, tensor<43x7x27xf32>, tensor<96x61x1x43xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 2 : i32} : (tensor<96x61x97x43xi1>) -> tensor<96x61x1x43xi1>
    %1 = tosa.pow %arg1, %arg2 : (tensor<43x7x27xf32>, tensor<1x1x27xf32>) -> tensor<43x7x27xf32>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<96x61x1x43xi1>, tensor<96x61x1x43xi1>) -> tensor<96x61x1x43xi1>
    %3 = tosa.arithmetic_right_shift %2, %2 {round = true} : (tensor<96x61x1x43xi1>, tensor<96x61x1x43xi1>) -> tensor<96x61x1x43xi1>
    %4 = tosa.bitwise_not %3 : (tensor<96x61x1x43xi1>) -> tensor<96x61x1x43xi1>
    %5 = tosa.logical_and %4, %0 : (tensor<96x61x1x43xi1>, tensor<96x61x1x43xi1>) -> tensor<96x61x1x43xi1>
    %6 = tosa.reduce_all %5 {axis = 1 : i32} : (tensor<96x61x1x43xi1>) -> tensor<96x1x1x43xi1>
    %7 = tosa.bitwise_xor %6, %6 : (tensor<96x1x1x43xi1>, tensor<96x1x1x43xi1>) -> tensor<96x1x1x43xi1>
    %8 = tosa.exp %1 : (tensor<43x7x27xf32>) -> tensor<43x7x27xf32>
    %9 = tosa.sub %8, %1 : (tensor<43x7x27xf32>, tensor<43x7x27xf32>) -> tensor<43x7x27xf32>
    %10 = tosa.sigmoid %9 : (tensor<43x7x27xf32>) -> tensor<43x7x27xf32>
    %11 = tosa.logical_not %2 : (tensor<96x61x1x43xi1>) -> tensor<96x61x1x43xi1>
    return %7, %10, %11 : tensor<96x1x1x43xi1>, tensor<43x7x27xf32>, tensor<96x61x1x43xi1>
  }
}
