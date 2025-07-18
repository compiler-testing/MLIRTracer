module {
  func.func @main(%arg0: tensor<93x6x91x8x27xi8>) -> tensor<93x6x91x8x27xi8> {
    %0 = tosa.identity %arg0 : (tensor<93x6x91x8x27xi8>) -> tensor<93x6x91x8x27xi8>
    return %0 : tensor<93x6x91x8x27xi8>
  }
}
