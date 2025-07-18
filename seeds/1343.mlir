module {
  func.func @main(%arg0: tensor<44xi8>, %arg1: tensor<44xi8>) -> tensor<44xi8> {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<44xi8>, tensor<44xi8>) -> tensor<44xi8>
    return %0 : tensor<44xi8>
  }
}
