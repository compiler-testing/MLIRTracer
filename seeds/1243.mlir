module {
  func.func @main(%arg0: tensor<32xf32>) -> (tensor<32xf32>, tensor<32xf32>, tensor<1x1xi1>) {
    %0 = tosa.rsqrt %arg0 : (tensor<32xf32>) -> tensor<32xf32>
    %1 = tosa.rsqrt %0 : (tensor<32xf32>) -> tensor<32xf32>
    %2 = tosa.floor %1 : (tensor<32xf32>) -> tensor<32xf32>
    %3 = tosa.equal %2, %2 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xi1>
    %4 = tosa.clz %3 : (tensor<32xi1>) -> tensor<32xi1>
    %5 = tosa.reverse %4 {axis = 0 : i32} : (tensor<32xi1>) -> tensor<32xi1>
    %6 = tosa.logical_right_shift %5, %5 : (tensor<32xi1>, tensor<32xi1>) -> tensor<32xi1>
    %7 = tosa.logical_left_shift %6, %5 : (tensor<32xi1>, tensor<32xi1>) -> tensor<32xi1>
    %8 = tosa.reduce_max %7 {axis = 0 : i32} : (tensor<32xi1>) -> tensor<1xi1>
    %9 = tosa.maximum %1, %2 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %r_10 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %10 = tosa.reshape %8, %r_10 : (tensor<1xi1>, !tosa.shape<2>) -> tensor<1x1xi1>
    %11 = tosa.pow %2, %1 : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %12 = tosa.reduce_sum %10 {axis = 1 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %13 = tosa.reduce_min %12 {axis = 0 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %14 = tosa.reduce_sum %13 {axis = 1 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    return %9, %11, %14 : tensor<32xf32>, tensor<32xf32>, tensor<1x1xi1>
  }
}
