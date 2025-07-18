module {
  func.func @main(%arg0: tensor<79x57x42xi1>) -> tensor<79x57x42xi1> {
    %0 = tosa.identity %arg0 : (tensor<79x57x42xi1>) -> tensor<79x57x42xi1>
    return %0 : tensor<79x57x42xi1>
  }
}
