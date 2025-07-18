module {
  func.func @main(%arg0: tensor<84x33x4x88xi8>, %arg1: tensor<1x1x4x1xi8>) -> tensor<84x33x4x88xi8> {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<84x33x4x88xi8>, tensor<1x1x4x1xi8>) -> tensor<84x33x4x88xi8>
    return %0 : tensor<84x33x4x88xi8>
  }
}
