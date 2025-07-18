module {
  func.func @main(%arg0: tensor<80x31x21x9x34xi32>, %arg1: tensor<1x31x1x9x1xi32>, %arg2: tensor<43x66xf32>, %arg3: tensor<1x66xf32>) -> (tensor<80x31x21x9x34xi1>, tensor<43x66xf32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<80x31x21x9x34xi32>, tensor<1x31x1x9x1xi32>) -> tensor<80x31x21x9x34xi1>
    %1 = tosa.pow %arg2, %arg3 : (tensor<43x66xf32>, tensor<1x66xf32>) -> tensor<43x66xf32>
    return %0, %1 : tensor<80x31x21x9x34xi1>, tensor<43x66xf32>
  }
}
