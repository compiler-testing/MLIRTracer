module {
  func.func @main(%arg0: tensor<89x71x77x89xf32>) -> tensor<89x71x77x89xf32> {
    %0 = tosa.sigmoid %arg0 : (tensor<89x71x77x89xf32>) -> tensor<89x71x77x89xf32>
    return %0 : tensor<89x71x77x89xf32>
  }
}
