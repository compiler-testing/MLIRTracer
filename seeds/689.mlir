module {
  func.func @main(%arg0: tensor<76xi1>) -> tensor<76xi1> {
    %0 = tosa.identity %arg0 : (tensor<76xi1>) -> tensor<76xi1>
    %1 = tosa.identity %0 : (tensor<76xi1>) -> tensor<76xi1>
    return %1 : tensor<76xi1>
  }
}
