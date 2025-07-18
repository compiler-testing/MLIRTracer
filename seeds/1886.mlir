module {
  func.func @main(%arg0: tensor<91x53x93x64x99x4xi64>, %arg1: tensor<1x53x93x1x1x4xi64>, %arg2: tensor<53xf32>) -> (tensor<91x53x93x64x99x4xi64>, tensor<53xf32>, tensor<53xi1>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<91x53x93x64x99x4xi64>, tensor<1x53x93x1x1x4xi64>) -> tensor<91x53x93x64x99x4xi64>
    %1 = tosa.sigmoid %arg2 : (tensor<53xf32>) -> tensor<53xf32>
    %2 = tosa.abs %1 : (tensor<53xf32>) -> tensor<53xf32>
    %3 = tosa.greater %2, %2 : (tensor<53xf32>, tensor<53xf32>) -> tensor<53xi1>
    %4 = tosa.exp %1 : (tensor<53xf32>) -> tensor<53xf32>
    %5 = tosa.abs %3 : (tensor<53xi1>) -> tensor<53xi1>
    return %0, %4, %5 : tensor<91x53x93x64x99x4xi64>, tensor<53xf32>, tensor<53xi1>
  }
}
