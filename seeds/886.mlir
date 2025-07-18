module {
  func.func @main(%arg0: tensor<23x30x1x33x92x32xi1>, %arg1: tensor<82xf32>) -> (tensor<23x30x1x33x92x32xi1>, tensor<1xi1>, tensor<164xf32>, tensor<164xf32>, tensor<1xi1>) {
    %0 = tosa.abs %arg0 : (tensor<23x30x1x33x92x32xi1>) -> tensor<23x30x1x33x92x32xi1>
    %t_1 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.tile %arg1, %t_1 : (tensor<82xf32>, !tosa.shape<1>) -> tensor<164xf32>
    %2 = tosa.bitwise_or %0, %0 : (tensor<23x30x1x33x92x32xi1>, tensor<23x30x1x33x92x32xi1>) -> tensor<23x30x1x33x92x32xi1>
    %3 = tosa.logical_left_shift %2, %0 : (tensor<23x30x1x33x92x32xi1>, tensor<23x30x1x33x92x32xi1>) -> tensor<23x30x1x33x92x32xi1>
    %4 = tosa.rsqrt %1 : (tensor<164xf32>) -> tensor<164xf32>
    %5 = tosa.reduce_min %4 {axis = 0 : i32} : (tensor<164xf32>) -> tensor<1xf32>
    %6 = tosa.tanh %5 : (tensor<1xf32>) -> tensor<1xf32>
    %7 = tosa.greater_equal %1, %4 : (tensor<164xf32>, tensor<164xf32>) -> tensor<164xi1>
    %8 = tosa.abs %6 : (tensor<1xf32>) -> tensor<1xf32>
    %9 = tosa.reduce_min %7 {axis = 0 : i32} : (tensor<164xi1>) -> tensor<1xi1>
    %10 = tosa.reduce_product %8 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %11 = tosa.greater %10, %5 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %12 = tosa.minimum %4, %1 : (tensor<164xf32>, tensor<164xf32>) -> tensor<164xf32>
    %13 = tosa.clz %9 : (tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.maximum %1, %4 : (tensor<164xf32>, tensor<164xf32>) -> tensor<164xf32>
    %15 = tosa.bitwise_not %13 : (tensor<1xi1>) -> tensor<1xi1>
    return %3, %11, %12, %14, %15 : tensor<23x30x1x33x92x32xi1>, tensor<1xi1>, tensor<164xf32>, tensor<164xf32>, tensor<1xi1>
  }
}
