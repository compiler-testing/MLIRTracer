module {
  func.func @main(%arg0: tensor<55x30x86xi8>, %arg1: tensor<55x30x1xi8>) -> tensor<55x30x86xi8> {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<55x30x86xi8>, tensor<55x30x1xi8>) -> tensor<55x30x86xi8>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<55x30x86xi8>, tensor<55x30x86xi8>) -> tensor<55x30x86xi8>
    return %1 : tensor<55x30x86xi8>
  }
}
