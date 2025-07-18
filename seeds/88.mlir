module {
  func.func @main(%arg0: tensor<96x35x51xi32>, %arg1: tensor<96x1x51xi32>, %arg2: tensor<7xf32>, %arg3: tensor<85x52x12xi1>) -> (tensor<21xf32>, tensor<11x9x6xi1>, tensor<85x52x12xi1>, tensor<96x35x51xi32>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<96x35x51xi32>, tensor<96x1x51xi32>) -> tensor<96x35x51xi32>
    %1 = tosa.maximum %0, %0 : (tensor<96x35x51xi32>, tensor<96x35x51xi32>) -> tensor<96x35x51xi32>
    %2 = tosa.rsqrt %arg2 : (tensor<7xf32>) -> tensor<7xf32>
    %t_3 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.tile %2, %t_3 : (tensor<7xf32>, !tosa.shape<1>) -> tensor<21xf32>
    %4 = tosa.logical_not %arg3 : (tensor<85x52x12xi1>) -> tensor<85x52x12xi1>
    %5 = tosa.bitwise_and %4, %4 : (tensor<85x52x12xi1>, tensor<85x52x12xi1>) -> tensor<85x52x12xi1>
    %6 = tosa.reciprocal %3 : (tensor<21xf32>) -> tensor<21xf32>
    %s_7_start = tosa.const_shape {values = dense<[ 58, 43, 6 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_7_size = tosa.const_shape {values = dense<[ 11, 9, 6 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = tosa.slice %5, %s_7_start, %s_7_size : (tensor<85x52x12xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<11x9x6xi1>
    %8 = tosa.clz %5 : (tensor<85x52x12xi1>) -> tensor<85x52x12xi1>
    %9 = tosa.bitwise_and %1, %0 : (tensor<96x35x51xi32>, tensor<96x35x51xi32>) -> tensor<96x35x51xi32>
    return %6, %7, %8, %9 : tensor<21xf32>, tensor<11x9x6xi1>, tensor<85x52x12xi1>, tensor<96x35x51xi32>
  }
}
