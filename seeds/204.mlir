module {
  func.func @main(%arg0: tensor<2x43x60x31xi32>, %arg1: tensor<69x66x87x54xf32>, %arg2: tensor<76x20x4xi1>, %arg3: tensor<1x20x4xi1>) -> (tensor<2x43x60x31xi32>, tensor<69x1x87x54xf32>, tensor<76x20x4xi1>) {
    %0 = tosa.reverse %arg0 {axis = 3 : i32} : (tensor<2x43x60x31xi32>) -> tensor<2x43x60x31xi32>
    %1 = tosa.ceil %arg1 : (tensor<69x66x87x54xf32>) -> tensor<69x66x87x54xf32>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<76x20x4xi1>, tensor<1x20x4xi1>) -> tensor<76x20x4xi1>
    %3 = tosa.log %1 : (tensor<69x66x87x54xf32>) -> tensor<69x66x87x54xf32>
    %4 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<69x66x87x54xf32>) -> tensor<69x1x87x54xf32>
    %5 = tosa.sigmoid %4 : (tensor<69x1x87x54xf32>) -> tensor<69x1x87x54xf32>
    %6 = tosa.logical_not %2 : (tensor<76x20x4xi1>) -> tensor<76x20x4xi1>
    return %0, %5, %6 : tensor<2x43x60x31xi32>, tensor<69x1x87x54xf32>, tensor<76x20x4xi1>
  }
}
