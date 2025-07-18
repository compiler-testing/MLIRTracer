module {
  func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %1 = tosa.identity %0 : (tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.identity %1 : (tensor<1xf32>) -> tensor<1xf32>
    return %2 : tensor<1xf32>
  }
}
