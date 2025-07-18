module {
  func.func @main(%arg0: tensor<93xi1>) -> tensor<1xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<93xi1>) -> tensor<1xi1>
    return %0 : tensor<1xi1>
  }
}
