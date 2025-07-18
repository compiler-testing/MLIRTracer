module {
  func.func @main(%arg0: tensor<48xf32>) -> tensor<i32> {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<48xf32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
