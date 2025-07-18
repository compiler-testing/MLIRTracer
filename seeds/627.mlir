module {
  func.func @main(%arg0: tensor<65x65xf32>) -> tensor<65x65xf32> {
    %0 = tosa.identity %arg0 : (tensor<65x65xf32>) -> tensor<65x65xf32>
    return %0 : tensor<65x65xf32>
  }
}
