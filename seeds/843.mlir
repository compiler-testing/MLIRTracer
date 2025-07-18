module {
  func.func @main(%arg0: tensor<90x74x13xf32>, %arg1: tensor<1x74x13xf32>) -> tensor<90x74x13xf32> {
    %0 = tosa.sub %arg0, %arg1 : (tensor<90x74x13xf32>, tensor<1x74x13xf32>) -> tensor<90x74x13xf32>
    return %0 : tensor<90x74x13xf32>
  }
}
