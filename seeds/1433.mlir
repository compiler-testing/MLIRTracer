module {
  func.func @main(%arg0: tensor<5x77x71x90xi16>) -> tensor<5x77x1x90xi16> {
    %0 = tosa.reduce_min %arg0 {axis = 2 : i32} : (tensor<5x77x71x90xi16>) -> tensor<5x77x1x90xi16>
    %1 = tosa.identity %0 : (tensor<5x77x1x90xi16>) -> tensor<5x77x1x90xi16>
    return %1 : tensor<5x77x1x90xi16>
  }
}
