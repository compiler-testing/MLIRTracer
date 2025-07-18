module {
  func.func @main(%arg0: tensor<78xf32>, %arg1: tensor<78xf32>, %arg2: tensor<6x5x81xf32>, %arg3: tensor<6x5x81xf32>) -> (tensor<78xi1>, tensor<6x5x81xi1>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<78xf32>, tensor<78xf32>) -> tensor<78xi1>
    %1 = tosa.sub %0, %0 : (tensor<78xi1>, tensor<78xi1>) -> tensor<78xi1>
    %2 = tosa.sub %1, %1 : (tensor<78xi1>, tensor<78xi1>) -> tensor<78xi1>
    %3 = tosa.logical_not %2 : (tensor<78xi1>) -> tensor<78xi1>
    %4 = tosa.maximum %arg2, %arg3 : (tensor<6x5x81xf32>, tensor<6x5x81xf32>) -> tensor<6x5x81xf32>
    %5 = tosa.equal %4, %4 : (tensor<6x5x81xf32>, tensor<6x5x81xf32>) -> tensor<6x5x81xi1>
    return %3, %5 : tensor<78xi1>, tensor<6x5x81xi1>
  }
}
