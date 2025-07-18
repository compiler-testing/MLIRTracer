module {
  func.func @main(%arg0: tensor<14x83x3x43x30xi8>, %arg1: tensor<14x39x3x43x30xi8>) -> tensor<14x122x3x43x30xi8> {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<14x83x3x43x30xi8>, tensor<14x39x3x43x30xi8>) -> tensor<14x122x3x43x30xi8>
    %1 = tosa.bitwise_or %0, %0 : (tensor<14x122x3x43x30xi8>, tensor<14x122x3x43x30xi8>) -> tensor<14x122x3x43x30xi8>
    return %1 : tensor<14x122x3x43x30xi8>
  }
}
