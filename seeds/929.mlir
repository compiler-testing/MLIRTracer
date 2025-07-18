module {
  func.func @main(%arg0: tensor<78x87x71x79xi1>, %arg1: tensor<60x13xf32>) -> (tensor<60x13xf32>, tensor<78x87x1x79xi1>) {
    %0 = tosa.logical_not %arg0 : (tensor<78x87x71x79xi1>) -> tensor<78x87x71x79xi1>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<78x87x71x79xi1>, tensor<78x87x71x79xi1>) -> tensor<78x87x71x79xi1>
    %2 = tosa.reciprocal %arg1 : (tensor<60x13xf32>) -> tensor<60x13xf32>
    %3 = tosa.reduce_any %1 {axis = 2 : i32} : (tensor<78x87x71x79xi1>) -> tensor<78x87x1x79xi1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<78x87x1x79xi1>, tensor<78x87x1x79xi1>) -> tensor<78x87x1x79xi1>
    return %2, %4 : tensor<60x13xf32>, tensor<78x87x1x79xi1>
  }
}
