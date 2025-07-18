module {
  func.func @main(%arg0: tensor<92xi8>) -> tensor<92xi8> {
    %0 = tosa.clz %arg0 : (tensor<92xi8>) -> tensor<92xi8>
    %1 = tosa.maximum %0, %0 : (tensor<92xi8>, tensor<92xi8>) -> tensor<92xi8>
    return %1 : tensor<92xi8>
  }
}
