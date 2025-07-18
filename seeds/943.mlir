module {
  func.func @main(%arg0: tensor<78xf32>) -> tensor<78xf32> {
    %0 = tosa.floor %arg0 : (tensor<78xf32>) -> tensor<78xf32>
    %1 = tosa.abs %0 : (tensor<78xf32>) -> tensor<78xf32>
    return %1 : tensor<78xf32>
  }
}
