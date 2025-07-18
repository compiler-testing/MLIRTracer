module {
  func.func @main(%arg0: tensor<3x86x91x45xf32>, %arg1: tensor<61xi1>) -> (tensor<3x86x91x45xf32>, tensor<3x86x1x45xi1>, tensor<1x86x1x45xi1>, tensor<3x86x1x45xf32>, tensor<3x86x91x45xf32>, tensor<1x1x1xi1>) {
    %0 = tosa.tanh %arg0 : (tensor<3x86x91x45xf32>) -> tensor<3x86x91x45xf32>
    %1 = tosa.reduce_min %0 {axis = 2 : i32} : (tensor<3x86x91x45xf32>) -> tensor<3x86x1x45xf32>
    %2 = tosa.add %1, %1 : (tensor<3x86x1x45xf32>, tensor<3x86x1x45xf32>) -> tensor<3x86x1x45xf32>
    %3 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<61xi1>) -> tensor<1xi1>
    %4 = tosa.greater_equal %2, %2 : (tensor<3x86x1x45xf32>, tensor<3x86x1x45xf32>) -> tensor<3x86x1x45xi1>
    %5 = tosa.ceil %0 : (tensor<3x86x91x45xf32>) -> tensor<3x86x91x45xf32>
    %6 = tosa.reduce_sum %4 {axis = 0 : i32} : (tensor<3x86x1x45xi1>) -> tensor<1x86x1x45xi1>
    %7 = tosa.rsqrt %2 : (tensor<3x86x1x45xf32>) -> tensor<3x86x1x45xf32>
    %8 = tosa.greater_equal %2, %7 : (tensor<3x86x1x45xf32>, tensor<3x86x1x45xf32>) -> tensor<3x86x1x45xi1>
    %9 = tosa.reduce_max %6 {axis = 0 : i32} : (tensor<1x86x1x45xi1>) -> tensor<1x86x1x45xi1>
    %10 = tosa.bitwise_and %3, %3 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %11 = tosa.sub %10, %10 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.log %2 : (tensor<3x86x1x45xf32>) -> tensor<3x86x1x45xf32>
    %13 = tosa.pow %0, %0 : (tensor<3x86x91x45xf32>, tensor<3x86x91x45xf32>) -> tensor<3x86x91x45xf32>
    %r_14 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %14 = tosa.reshape %11, %r_14 : (tensor<1xi1>, !tosa.shape<3>) -> tensor<1x1x1xi1>
    return %5, %8, %9, %12, %13, %14 : tensor<3x86x91x45xf32>, tensor<3x86x1x45xi1>, tensor<1x86x1x45xi1>, tensor<3x86x1x45xf32>, tensor<3x86x91x45xf32>, tensor<1x1x1xi1>
  }
}
