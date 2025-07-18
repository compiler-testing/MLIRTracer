module {
  func.func @main(%arg0: tensor<88x96xi8>) -> tensor<88x192xi8> {
    %0 = tosa.abs %arg0 : (tensor<88x96xi8>) -> tensor<88x96xi8>
    %1 = tosa.bitwise_or %0, %0 : (tensor<88x96xi8>, tensor<88x96xi8>) -> tensor<88x96xi8>
    %2 = tosa.reverse %1 {axis = 0 : i32} : (tensor<88x96xi8>) -> tensor<88x96xi8>
    %3 = tosa.reverse %2 {axis = 0 : i32} : (tensor<88x96xi8>) -> tensor<88x96xi8>
    %4 = tosa.reverse %3 {axis = 1 : i32} : (tensor<88x96xi8>) -> tensor<88x96xi8>
    %5 = tosa.concat %4, %0 {axis = 1 : i32} : (tensor<88x96xi8>, tensor<88x96xi8>) -> tensor<88x192xi8>
    return %5 : tensor<88x192xi8>
  }
}
