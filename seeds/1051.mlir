module {
  func.func @main(%arg0: tensor<17x74x27x80xi1>) -> tensor<17x74x27x1xi1> {
    %0 = tosa.reduce_sum %arg0 {axis = 3 : i32} : (tensor<17x74x27x80xi1>) -> tensor<17x74x27x1xi1>
    return %0 : tensor<17x74x27x1xi1>
  }
}
