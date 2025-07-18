module {
  func.func @main(%arg0: tensor<71xi1>) -> tensor<1xi1> {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<71xi1>) -> tensor<1xi1>
    return %0 : tensor<1xi1>
  }
}
