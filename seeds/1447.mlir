module {
  func.func @main(%arg0: tensor<55x84x99x49x75x36xf32>) -> tensor<55x84x99x49x75x36xf32> {
    %0 = tosa.ceil %arg0 : (tensor<55x84x99x49x75x36xf32>) -> tensor<55x84x99x49x75x36xf32>
    return %0 : tensor<55x84x99x49x75x36xf32>
  }
}
