module {
  func.func @main(%arg0: tensor<74xi32>) -> tensor<i32> {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<74xi32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
