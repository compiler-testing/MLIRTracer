module {
  func.func @main(%arg0: tensor<90x20x25x6x33xf32>, %arg1: tensor<1x20x25x6x33xf32>) -> tensor<90x20x25x6x33xf32> {
    %0 = tosa.pow %arg0, %arg1 : (tensor<90x20x25x6x33xf32>, tensor<1x20x25x6x33xf32>) -> tensor<90x20x25x6x33xf32>
    %1 = tosa.maximum %0, %0 : (tensor<90x20x25x6x33xf32>, tensor<90x20x25x6x33xf32>) -> tensor<90x20x25x6x33xf32>
    return %1 : tensor<90x20x25x6x33xf32>
  }
}
