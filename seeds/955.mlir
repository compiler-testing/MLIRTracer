module {
  func.func @main(%arg0: tensor<59x97x38x53xi1>, %arg1: tensor<59x97x53x53xi1>) -> tensor<59x97x91x53xi1> {
    %0 = tosa.concat %arg0, %arg1 {axis = 2 : i32} : (tensor<59x97x38x53xi1>, tensor<59x97x53x53xi1>) -> tensor<59x97x91x53xi1>
    return %0 : tensor<59x97x91x53xi1>
  }
}
