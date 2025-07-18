module {
  func.func @main(%arg0: tensor<77xf32>) -> tensor<1xf32> {
    %0 = tosa.reciprocal %arg0 : (tensor<77xf32>) -> tensor<77xf32>
    %1 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<77xf32>) -> tensor<1xf32>
    %2 = tosa.maximum %1, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %2 : tensor<1xf32>
  }
}
