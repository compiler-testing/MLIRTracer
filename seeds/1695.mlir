module {
  func.func @main(%arg0: tensor<16xi1>, %arg1: tensor<86x100x92x69x22x64xf32>) -> (tensor<1xi1>, tensor<1xi1>, tensor<3xi1>, tensor<1xi1>, tensor<1xi1>, tensor<86x100x92x69x22x64xf32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<16xi1>) -> tensor<1xi1>
    %1 = tosa.add %0, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.floor %arg1 : (tensor<86x100x92x69x22x64xf32>) -> tensor<86x100x92x69x22x64xf32>
    %3 = tosa.tanh %2 : (tensor<86x100x92x69x22x64xf32>) -> tensor<86x100x92x69x22x64xf32>
    %4 = tosa.reverse %1 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.reciprocal %3 : (tensor<86x100x92x69x22x64xf32>) -> tensor<86x100x92x69x22x64xf32>
    %6 = tosa.bitwise_xor %1, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %t_7 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %7 = tosa.tile %0, %t_7 : (tensor<1xi1>, !tosa.shape<1>) -> tensor<3xi1>
    %8 = tosa.bitwise_not %1 : (tensor<1xi1>) -> tensor<1xi1>
    %9 = tosa.exp %5 : (tensor<86x100x92x69x22x64xf32>) -> tensor<86x100x92x69x22x64xf32>
    %10 = tosa.add %9, %3 : (tensor<86x100x92x69x22x64xf32>, tensor<86x100x92x69x22x64xf32>) -> tensor<86x100x92x69x22x64xf32>
    %11 = tosa.logical_or %1, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.reverse %8 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %13 = tosa.rsqrt %10 : (tensor<86x100x92x69x22x64xf32>) -> tensor<86x100x92x69x22x64xf32>
    return %4, %6, %7, %11, %12, %13 : tensor<1xi1>, tensor<1xi1>, tensor<3xi1>, tensor<1xi1>, tensor<1xi1>, tensor<86x100x92x69x22x64xf32>
  }
}
