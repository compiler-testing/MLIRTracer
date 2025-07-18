module {
  func.func @main(%arg0: tensor<44xf32>, %arg1: tensor<1xf32>) -> tensor<44xi1> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<44xf32>, tensor<1xf32>) -> tensor<44xf32>
    %1 = tosa.ceil %0 : (tensor<44xf32>) -> tensor<44xf32>
    %2 = tosa.greater_equal %1, %0 : (tensor<44xf32>, tensor<44xf32>) -> tensor<44xi1>
    return %2 : tensor<44xi1>
  }
}
