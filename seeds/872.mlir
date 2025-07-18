module {
  func.func @main(%arg0: tensor<90x21xi1>, %arg1: tensor<62x3x22x32x52xf32>) -> (tensor<90x21xi1>, tensor<62x3x22x32x52xf32>) {
    %0 = tosa.identity %arg0 : (tensor<90x21xi1>) -> tensor<90x21xi1>
    %1 = tosa.exp %arg1 : (tensor<62x3x22x32x52xf32>) -> tensor<62x3x22x32x52xf32>
    return %0, %1 : tensor<90x21xi1>, tensor<62x3x22x32x52xf32>
  }
}
