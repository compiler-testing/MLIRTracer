module {
  func.func @main(%arg0: tensor<77xf32>) -> tensor<77xf32> {
    %0 = tosa.log %arg0 : (tensor<77xf32>) -> tensor<77xf32>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<77xf32>) -> tensor<77xf32>
    return %1 : tensor<77xf32>
  }
}
