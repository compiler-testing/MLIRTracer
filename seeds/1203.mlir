module {
  func.func @main(%arg0: tensor<95xi8>) -> tensor<95xi8> {
    %0 = tosa.clz %arg0 : (tensor<95xi8>) -> tensor<95xi8>
    return %0 : tensor<95xi8>
  }
}
