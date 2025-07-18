module {
  func.func @main(%arg0: tensor<46x13x27xi8>, %arg1: tensor<1x13x1xi8>) -> tensor<46x13x27xi8> {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<46x13x27xi8>, tensor<1x13x1xi8>) -> tensor<46x13x27xi8>
    %1 = tosa.abs %0 : (tensor<46x13x27xi8>) -> tensor<46x13x27xi8>
    return %1 : tensor<46x13x27xi8>
  }
}
