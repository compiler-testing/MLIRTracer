module {
  func.func @main(%arg0: tensor<64xf32>, %arg1: tensor<81xi1>, %arg2: tensor<81xi1>) -> (tensor<81xi1>, tensor<64xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<64xf32>) -> tensor<64xf32>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<81xi1>, tensor<81xi1>) -> tensor<81xi1>
    %2 = tosa.greater_equal %0, %0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xi1>
    return %1, %2 : tensor<81xi1>, tensor<64xi1>
  }
}
