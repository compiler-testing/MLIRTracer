module {
  func.func @main(%arg0: tensor<85x1x69x14xi1>, %arg1: tensor<1x1x69x1xi1>, %arg2: tensor<30x99x5x32x80xf32>) -> (tensor<30x99x5x32x80xf32>, tensor<1x1x28xi32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<85x1x69x14xi1>, tensor<1x1x69x1xi1>) -> tensor<85x1x69x14xi1>
    %1 = tosa.reduce_all %0 {axis = 0 : i32} : (tensor<85x1x69x14xi1>) -> tensor<1x1x69x14xi1>
    %2 = tosa.log %arg2 : (tensor<30x99x5x32x80xf32>) -> tensor<30x99x5x32x80xf32>
    %3 = tosa.argmax %1 {axis = 1 : i32} : (tensor<1x1x69x14xi1>) -> tensor<1x69x14xi32>
    %4 = tosa.concat %3, %3 {axis = 2 : i32} : (tensor<1x69x14xi32>, tensor<1x69x14xi32>) -> tensor<1x69x28xi32>
    %5 = tosa.reduce_product %4 {axis = 1 : i32} : (tensor<1x69x28xi32>) -> tensor<1x1x28xi32>
    return %2, %5 : tensor<30x99x5x32x80xf32>, tensor<1x1x28xi32>
  }
}
