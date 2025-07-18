module {
  func.func @main(%arg0: tensor<20x36x73xi1>) -> tensor<1x36xi32> {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<20x36x73xi1>) -> tensor<1x36x73xi1>
    %1 = tosa.argmax %0 {axis = 2 : i32} : (tensor<1x36x73xi1>) -> tensor<1x36xi32>
    return %1 : tensor<1x36xi32>
  }
}
