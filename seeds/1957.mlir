module {
  func.func @main(%arg0: tensor<34x81x41x15x65x33xi8>, %arg1: tensor<28x9xi1>, %arg2: tensor<28x9xi1>) -> (tensor<28x9xi1>, tensor<34x81x41x15x65x33xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<34x81x41x15x65x33xi8>) -> tensor<34x81x41x15x65x33xi8>
    %1 = tosa.logical_and %arg1, %arg2 : (tensor<28x9xi1>, tensor<28x9xi1>) -> tensor<28x9xi1>
    %2 = tosa.equal %0, %0 : (tensor<34x81x41x15x65x33xi8>, tensor<34x81x41x15x65x33xi8>) -> tensor<34x81x41x15x65x33xi1>
    return %1, %2 : tensor<28x9xi1>, tensor<34x81x41x15x65x33xi1>
  }
}
