module {
  func.func @main(%arg0: tensor<60x71x28x86xi1>) -> tensor<1x71x28x86xi1> {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<60x71x28x86xi1>) -> tensor<1x71x28x86xi1>
    return %0 : tensor<1x71x28x86xi1>
  }
}
