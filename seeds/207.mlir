module {
  func.func @main(%arg0: tensor<65xf32>) -> tensor<65xf32> {
    %0 = tosa.floor %arg0 : (tensor<65xf32>) -> tensor<65xf32>
    %1 = tosa.minimum %0, %0 : (tensor<65xf32>, tensor<65xf32>) -> tensor<65xf32>
    return %1 : tensor<65xf32>
  }
}
