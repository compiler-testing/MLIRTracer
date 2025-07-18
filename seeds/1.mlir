module {
  func.func @main(%arg0: tensor<43xi1>, %arg1: tensor<72x24x86xi8>, %arg2: tensor<1x24x86xi8>) -> (tensor<43xi1>, tensor<72x24x86xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<43xi1>) -> tensor<43xi1>
    %1 = tosa.equal %arg1, %arg2 : (tensor<72x24x86xi8>, tensor<1x24x86xi8>) -> tensor<72x24x86xi1>
    return %0, %1 : tensor<43xi1>, tensor<72x24x86xi1>
  }
}
