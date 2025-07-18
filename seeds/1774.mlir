module {
  func.func @main(%arg0: tensor<79x2x42x58xf32>) -> tensor<79x2x42x58xf32> {
    %0 = tosa.identity %arg0 : (tensor<79x2x42x58xf32>) -> tensor<79x2x42x58xf32>
    return %0 : tensor<79x2x42x58xf32>
  }
}
