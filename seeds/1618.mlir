module {
  func.func @main(%arg0: tensor<54x76xf32>, %arg1: tensor<45x68x33xi1>) -> (tensor<54x1xf32>, tensor<1x68x33xi1>, tensor<54x1xf32>, tensor<2244xi1>) {
    %0 = tosa.reduce_product %arg0 {axis = 1 : i32} : (tensor<54x76xf32>) -> tensor<54x1xf32>
    %1 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<45x68x33xi1>) -> tensor<1x68x33xi1>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<1x68x33xi1>) -> tensor<1x68x33xi1>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<1x68x33xi1>, tensor<1x68x33xi1>) -> tensor<1x68x33xi1>
    %4 = tosa.maximum %0, %0 : (tensor<54x1xf32>, tensor<54x1xf32>) -> tensor<54x1xf32>
    %5 = tosa.arithmetic_right_shift %3, %2 {round = false} : (tensor<1x68x33xi1>, tensor<1x68x33xi1>) -> tensor<1x68x33xi1>
    %6 = tosa.logical_left_shift %5, %2 : (tensor<1x68x33xi1>, tensor<1x68x33xi1>) -> tensor<1x68x33xi1>
    %7 = tosa.logical_not %6 : (tensor<1x68x33xi1>) -> tensor<1x68x33xi1>
    %8 = tosa.reciprocal %0 : (tensor<54x1xf32>) -> tensor<54x1xf32>
    %9 = tosa.logical_and %7, %6 : (tensor<1x68x33xi1>, tensor<1x68x33xi1>) -> tensor<1x68x33xi1>
    %r_10 = tosa.const_shape {values = dense<[ 2244 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %10 = tosa.reshape %2, %r_10 : (tensor<1x68x33xi1>, !tosa.shape<1>) -> tensor<2244xi1>
    %11 = tosa.bitwise_or %10, %10 : (tensor<2244xi1>, tensor<2244xi1>) -> tensor<2244xi1>
    %12 = tosa.ceil %4 : (tensor<54x1xf32>) -> tensor<54x1xf32>
    %13 = tosa.arithmetic_right_shift %11, %11 {round = false} : (tensor<2244xi1>, tensor<2244xi1>) -> tensor<2244xi1>
    return %8, %9, %12, %13 : tensor<54x1xf32>, tensor<1x68x33xi1>, tensor<54x1xf32>, tensor<2244xi1>
  }
}
