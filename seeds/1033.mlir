module {
  func.func @main(%arg0: tensor<43xi32>, %arg1: tensor<1xi32>) -> tensor<43xi32> {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<43xi32>, tensor<1xi32>) -> tensor<43xi32>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<43xi32>, tensor<43xi32>) -> tensor<43xi32>
    %2 = tosa.clz %1 : (tensor<43xi32>) -> tensor<43xi32>
    return %2 : tensor<43xi32>
  }
}
