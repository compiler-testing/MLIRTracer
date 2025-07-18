module {
  func.func @main(%arg0: tensor<29xi8>) -> tensor<1xi8> {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<29xi8>) -> tensor<1xi8>
    return %0 : tensor<1xi8>
  }
}
