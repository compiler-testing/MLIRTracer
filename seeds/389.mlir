module {
  func.func @main(%arg0: tensor<70x36xi16>, %arg1: tensor<88x23xi1>) -> (tensor<70x36xi16>, tensor<88x1xi1>) {
    %0 = tosa.abs %arg0 : (tensor<70x36xi16>) -> tensor<70x36xi16>
    %1 = tosa.sub %0, %0 : (tensor<70x36xi16>, tensor<70x36xi16>) -> tensor<70x36xi16>
    %2 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<88x23xi1>) -> tensor<88x1xi1>
    %3 = tosa.logical_and %2, %2 : (tensor<88x1xi1>, tensor<88x1xi1>) -> tensor<88x1xi1>
    %4 = tosa.logical_or %3, %3 : (tensor<88x1xi1>, tensor<88x1xi1>) -> tensor<88x1xi1>
    return %1, %4 : tensor<70x36xi16>, tensor<88x1xi1>
  }
}
