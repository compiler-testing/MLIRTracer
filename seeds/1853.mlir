module {
  func.func @main(%arg0: tensor<61x53xi1>, %arg1: tensor<82xf32>) -> (tensor<61x53xi1>, tensor<82xf32>) {
    %0 = tosa.abs %arg0 : (tensor<61x53xi1>) -> tensor<61x53xi1>
    %1 = tosa.log %arg1 : (tensor<82xf32>) -> tensor<82xf32>
    return %0, %1 : tensor<61x53xi1>, tensor<82xf32>
  }
}
