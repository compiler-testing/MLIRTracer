module {
  func.func @main(%arg0: tensor<34x26xi32>, %arg1: tensor<86x96x32x87x28x3xf32>) -> (tensor<86x96x32x87x28x3xf32>, tensor<34x1xi32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<34x26xi32>) -> tensor<34x1xi32>
    %1 = tosa.log %arg1 : (tensor<86x96x32x87x28x3xf32>) -> tensor<86x96x32x87x28x3xf32>
    %2 = tosa.rsqrt %1 : (tensor<86x96x32x87x28x3xf32>) -> tensor<86x96x32x87x28x3xf32>
    %3 = tosa.bitwise_and %0, %0 : (tensor<34x1xi32>, tensor<34x1xi32>) -> tensor<34x1xi32>
    return %2, %3 : tensor<86x96x32x87x28x3xf32>, tensor<34x1xi32>
  }
}
