module {
  func.func @main(%arg0: tensor<3xi8>) -> tensor<3xi8> {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<3xi8>) -> tensor<3xi8>
    return %0 : tensor<3xi8>
  }
}
