module {
  func.func @main(%arg0: tensor<71x67x45xf32>) -> tensor<71x67x45xf32> {
    %0 = tosa.abs %arg0 : (tensor<71x67x45xf32>) -> tensor<71x67x45xf32>
    %1 = tosa.sigmoid %0 : (tensor<71x67x45xf32>) -> tensor<71x67x45xf32>
    return %1 : tensor<71x67x45xf32>
  }
}
