module {
  func.func @main(%arg0: tensor<44x6x7xi8>, %arg1: tensor<1x1x7xi8>) -> tensor<88x6x7xi8> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<44x6x7xi8>, tensor<1x1x7xi8>) -> tensor<44x6x7xi8>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<44x6x7xi8>, tensor<44x6x7xi8>) -> tensor<88x6x7xi8>
    return %1 : tensor<88x6x7xi8>
  }
}
