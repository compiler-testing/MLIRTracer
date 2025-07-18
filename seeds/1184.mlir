module {
  func.func @main(%arg0: tensor<2x49x35x25xi1>, %arg1: tensor<1x1x1x25xi1>, %arg2: tensor<97x40x33x77x14xf32>) -> (tensor<2x49x35x25xi1>, tensor<2x1x35x25xi1>, tensor<97x40x33x77x14xf32>, tensor<2x49x1x25xi1>, tensor<2x49x1x25xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<2x49x35x25xi1>, tensor<1x1x1x25xi1>) -> tensor<2x49x35x25xi1>
    %1 = tosa.ceil %arg2 : (tensor<97x40x33x77x14xf32>) -> tensor<97x40x33x77x14xf32>
    %2 = tosa.logical_or %0, %0 : (tensor<2x49x35x25xi1>, tensor<2x49x35x25xi1>) -> tensor<2x49x35x25xi1>
    %3 = tosa.sub %2, %0 : (tensor<2x49x35x25xi1>, tensor<2x49x35x25xi1>) -> tensor<2x49x35x25xi1>
    %4 = tosa.pow %1, %1 : (tensor<97x40x33x77x14xf32>, tensor<97x40x33x77x14xf32>) -> tensor<97x40x33x77x14xf32>
    %5 = tosa.reduce_product %2 {axis = 1 : i32} : (tensor<2x49x35x25xi1>) -> tensor<2x1x35x25xi1>
    %6 = tosa.log %1 : (tensor<97x40x33x77x14xf32>) -> tensor<97x40x33x77x14xf32>
    %7 = tosa.reverse %5 {axis = 1 : i32} : (tensor<2x1x35x25xi1>) -> tensor<2x1x35x25xi1>
    %8 = tosa.sub %4, %6 : (tensor<97x40x33x77x14xf32>, tensor<97x40x33x77x14xf32>) -> tensor<97x40x33x77x14xf32>
    %9 = tosa.reduce_all %2 {axis = 2 : i32} : (tensor<2x49x35x25xi1>) -> tensor<2x49x1x25xi1>
    %10 = tosa.reduce_min %0 {axis = 2 : i32} : (tensor<2x49x35x25xi1>) -> tensor<2x49x1x25xi1>
    return %3, %7, %8, %9, %10 : tensor<2x49x35x25xi1>, tensor<2x1x35x25xi1>, tensor<97x40x33x77x14xf32>, tensor<2x49x1x25xi1>, tensor<2x49x1x25xi1>
  }
}
