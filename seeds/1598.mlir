module {
  func.func @main(%arg0: tensor<64xf32>) -> tensor<64xf32> {
    %0 = tosa.exp %arg0 : (tensor<64xf32>) -> tensor<64xf32>
    %1 = tosa.exp %0 : (tensor<64xf32>) -> tensor<64xf32>
    %2 = tosa.ceil %1 : (tensor<64xf32>) -> tensor<64xf32>
    return %2 : tensor<64xf32>
  }
}
