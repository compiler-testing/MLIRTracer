module {
  func.func @main(%arg0: tensor<76x78x18x2x73xi1>, %arg1: tensor<12x32x59x80x46xf32>, %arg2: tensor<28xi8>, %arg3: tensor<16x74x25xi1>) -> (tensor<76x78x18x2x73xi1>, tensor<12x32x59x80x46xi1>, tensor<1xi8>, tensor<16x74x1xi1>, tensor<11x11x2x7x5xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<76x78x18x2x73xi1>) -> tensor<76x78x18x2x73xi1>
    %1 = tosa.tanh %arg1 : (tensor<12x32x59x80x46xf32>) -> tensor<12x32x59x80x46xf32>
    %2 = tosa.greater_equal %1, %1 : (tensor<12x32x59x80x46xf32>, tensor<12x32x59x80x46xf32>) -> tensor<12x32x59x80x46xi1>
    %3 = tosa.log %1 : (tensor<12x32x59x80x46xf32>) -> tensor<12x32x59x80x46xf32>
    %4 = tosa.reduce_max %arg2 {axis = 0 : i32} : (tensor<28xi8>) -> tensor<1xi8>
    %5 = tosa.reduce_all %arg3 {axis = 2 : i32} : (tensor<16x74x25xi1>) -> tensor<16x74x1xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 1, 11, 3, 5, 8 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_6_size = tosa.const_shape {values = dense<[ 11, 11, 2, 7, 5 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %6 = tosa.slice %3, %s_6_start, %s_6_size : (tensor<12x32x59x80x46xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<11x11x2x7x5xf32>
    return %0, %2, %4, %5, %6 : tensor<76x78x18x2x73xi1>, tensor<12x32x59x80x46xi1>, tensor<1xi8>, tensor<16x74x1xi1>, tensor<11x11x2x7x5xf32>
  }
}
