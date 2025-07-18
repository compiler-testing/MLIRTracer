module {
  func.func @main(%arg0: tensor<2xi1>) -> tensor<1xi1> {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<2xi1>) -> tensor<1xi1>
    return %0 : tensor<1xi1>
  }
}
