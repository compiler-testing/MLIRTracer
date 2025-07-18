module {
  func.func @main(%arg0: tensor<1x32x7x40xi32>, %arg1: tensor<12xf32>) -> (tensor<12xf32>, tensor<1x32x7x40xi32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<1x32x7x40xi32>) -> tensor<1x32x7x40xi32>
    %1 = tosa.bitwise_or %0, %0 : (tensor<1x32x7x40xi32>, tensor<1x32x7x40xi32>) -> tensor<1x32x7x40xi32>
    %2 = tosa.log %arg1 : (tensor<12xf32>) -> tensor<12xf32>
    %3 = tosa.maximum %2, %2 : (tensor<12xf32>, tensor<12xf32>) -> tensor<12xf32>
    %4 = tosa.bitwise_and %1, %0 : (tensor<1x32x7x40xi32>, tensor<1x32x7x40xi32>) -> tensor<1x32x7x40xi32>
    return %3, %4 : tensor<12xf32>, tensor<1x32x7x40xi32>
  }
}
