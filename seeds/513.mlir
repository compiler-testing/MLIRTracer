module {
  func.func @main(%arg0: tensor<57x4xf32>, %arg1: tensor<88x80x68x53x34x3xi1>, %arg2: tensor<88x80x1x1x1x1xi1>) -> (tensor<88x80x68x53x34x3xi1>, tensor<1x4xi1>, tensor<1x4xf32>) {
    %0 = tosa.exp %arg0 : (tensor<57x4xf32>) -> tensor<57x4xf32>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<57x4xf32>) -> tensor<1x4xf32>
    %2 = tosa.floor %1 : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %3 = tosa.logical_or %arg1, %arg2 : (tensor<88x80x68x53x34x3xi1>, tensor<88x80x1x1x1x1xi1>) -> tensor<88x80x68x53x34x3xi1>
    %4 = tosa.sigmoid %2 : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %5 = tosa.greater %4, %4 : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xi1>
    %6 = tosa.sigmoid %2 : (tensor<1x4xf32>) -> tensor<1x4xf32>
    return %3, %5, %6 : tensor<88x80x68x53x34x3xi1>, tensor<1x4xi1>, tensor<1x4xf32>
  }
}
