module {
  func.func @main(%arg0: tensor<84x84xi8>) -> tensor<84x1xi8> {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<84x84xi8>) -> tensor<84x1xi8>
    return %0 : tensor<84x1xi8>
  }
}
