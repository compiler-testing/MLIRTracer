module {
  func.func @main(%arg0: tensor<86xf32>, %arg1: tensor<86xf32>) -> tensor<86xf32> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<86xf32>, tensor<86xf32>) -> tensor<86xf32>
    %1 = tosa.add %0, %0 : (tensor<86xf32>, tensor<86xf32>) -> tensor<86xf32>
    %2 = tosa.abs %1 : (tensor<86xf32>) -> tensor<86xf32>
    return %2 : tensor<86xf32>
  }
}
