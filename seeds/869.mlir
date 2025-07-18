module {
  func.func @main(%arg0: tensor<44x65x18xi8>, %arg1: tensor<1x65x18xi8>) -> tensor<44x65x18xi8> {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<44x65x18xi8>, tensor<1x65x18xi8>) -> tensor<44x65x18xi8>
    return %0 : tensor<44x65x18xi8>
  }
}
