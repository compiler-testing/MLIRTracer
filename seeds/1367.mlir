module {
  func.func @main(%arg0: tensor<78x25x82xi1>, %arg1: tensor<76x7x77xf32>) -> (tensor<78x1xi1>, tensor<76x7x77xf32>, tensor<78x1x82xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<78x25x82xi1>) -> tensor<78x1x82xi1>
    %1 = tosa.argmax %0 {axis = 2 : i32} : (tensor<78x1x82xi1>) -> tensor<78x1xi32>
    %2 = tosa.greater %1, %1 : (tensor<78x1xi32>, tensor<78x1xi32>) -> tensor<78x1xi1>
    %3 = tosa.sigmoid %arg1 : (tensor<76x7x77xf32>) -> tensor<76x7x77xf32>
    %4 = tosa.log %3 : (tensor<76x7x77xf32>) -> tensor<76x7x77xf32>
    %5 = tosa.logical_not %0 : (tensor<78x1x82xi1>) -> tensor<78x1x82xi1>
    return %2, %4, %5 : tensor<78x1xi1>, tensor<76x7x77xf32>, tensor<78x1x82xi1>
  }
}
