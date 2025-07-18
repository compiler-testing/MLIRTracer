module {
  func.func @main(%arg0: tensor<49xi32>, %arg1: tensor<1xi32>) -> tensor<49xi1> {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<49xi32>, tensor<1xi32>) -> tensor<49xi1>
    %1 = tosa.identity %0 : (tensor<49xi1>) -> tensor<49xi1>
    return %1 : tensor<49xi1>
  }
}
