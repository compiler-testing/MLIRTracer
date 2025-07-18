module {
  func.func @main(%arg0: tensor<31x27x53xi1>) -> tensor<1x27x53xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<31x27x53xi1>) -> tensor<1x27x53xi1>
    return %0 : tensor<1x27x53xi1>
  }
}
