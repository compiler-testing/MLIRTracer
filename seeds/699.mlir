module {
  func.func @main(%arg0: tensor<61x42x59x20x2x14xi1>, %arg1: tensor<1x42x1x20x1x14xi1>, %arg2: tensor<41xi64>, %arg3: tensor<1xi64>, %arg4: tensor<83xf32>) -> (tensor<1x854x1xi1>, tensor<83xf32>, tensor<10xi1>, tensor<10xi64>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<61x42x59x20x2x14xi1>, tensor<1x42x1x20x1x14xi1>) -> tensor<61x42x59x20x2x14xi1>
    %1 = tosa.sub %0, %0 : (tensor<61x42x59x20x2x14xi1>, tensor<61x42x59x20x2x14xi1>) -> tensor<61x42x59x20x2x14xi1>
    %r_2 = tosa.const_shape {values = dense<[ 1, 854, 99120 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.reshape %1, %r_2 : (tensor<61x42x59x20x2x14xi1>, !tosa.shape<3>) -> tensor<1x854x99120xi1>
    %3 = tosa.minimum %arg2, %arg3 : (tensor<41xi64>, tensor<1xi64>) -> tensor<41xi64>
    %4 = tosa.reduce_sum %2 {axis = 2 : i32} : (tensor<1x854x99120xi1>) -> tensor<1x854x1xi1>
    %r_5 = tosa.const_shape {values = dense<[ 1, 854, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.reshape %4, %r_5 : (tensor<1x854x1xi1>, !tosa.shape<3>) -> tensor<1x854x1xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_6_size = tosa.const_shape {values = dense<[ 10 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.slice %3, %s_6_start, %s_6_size : (tensor<41xi64>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<10xi64>
    %7 = tosa.greater_equal %6, %6 : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi1>
    %8 = tosa.ceil %arg4 : (tensor<83xf32>) -> tensor<83xf32>
    %9 = tosa.arithmetic_right_shift %6, %6 {round = false} : (tensor<10xi64>, tensor<10xi64>) -> tensor<10xi64>
    %r_10 = tosa.const_shape {values = dense<[ 10 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %10 = tosa.reshape %7, %r_10 : (tensor<10xi1>, !tosa.shape<1>) -> tensor<10xi1>
    %11 = tosa.logical_not %10 : (tensor<10xi1>) -> tensor<10xi1>
    %12 = tosa.abs %9 : (tensor<10xi64>) -> tensor<10xi64>
    return %5, %8, %11, %12 : tensor<1x854x1xi1>, tensor<83xf32>, tensor<10xi1>, tensor<10xi64>
  }
}
