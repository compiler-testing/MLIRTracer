module {
  func.func @main(%arg0: tensor<91x80x98x44x85xf32>) -> tensor<91x80x98x44x85xf32> {
    %0 = tosa.ceil %arg0 : (tensor<91x80x98x44x85xf32>) -> tensor<91x80x98x44x85xf32>
    return %0 : tensor<91x80x98x44x85xf32>
  }
}
