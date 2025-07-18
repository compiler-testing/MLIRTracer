module {
  func.func @main(%arg0: tensor<92xi8>, %arg1: tensor<1xi8>) -> tensor<92xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<92xi8>, tensor<1xi8>) -> tensor<92xi1>
    return %0 : tensor<92xi1>
  }
}
