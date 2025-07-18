module {
  func.func @main(%arg0: tensor<57x20x89xi8>, %arg1: tensor<57x1x89xi8>) -> tensor<57x20x89xi8> {
    %0 = tosa.add %arg0, %arg1 : (tensor<57x20x89xi8>, tensor<57x1x89xi8>) -> tensor<57x20x89xi8>
    return %0 : tensor<57x20x89xi8>
  }
}
