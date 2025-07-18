module {
  func.func @main(%arg0: tensor<10x7xi8>, %arg1: tensor<10x1xi8>) -> (tensor<10x7xi8>, tensor<10x7xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<10x7xi8>, tensor<10x1xi8>) -> tensor<10x7xi8>
    %1 = tosa.greater_equal %0, %0 : (tensor<10x7xi8>, tensor<10x7xi8>) -> tensor<10x7xi1>
    %2 = tosa.maximum %0, %0 : (tensor<10x7xi8>, tensor<10x7xi8>) -> tensor<10x7xi8>
    %3 = tosa.logical_xor %1, %1 : (tensor<10x7xi1>, tensor<10x7xi1>) -> tensor<10x7xi1>
    return %2, %3 : tensor<10x7xi8>, tensor<10x7xi1>
  }
}
