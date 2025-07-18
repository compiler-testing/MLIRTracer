module {
  func.func @main(%arg0: tensor<94x32x95x98x1x12xi64>, %arg1: tensor<1x1x95x98x1x1xi64>, %arg2: tensor<55x10xi32>, %arg3: tensor<1x1xi32>) -> (tensor<94x32x95x98x1x12xi1>, tensor<55x10xi1>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<94x32x95x98x1x12xi64>, tensor<1x1x95x98x1x1xi64>) -> tensor<94x32x95x98x1x12xi1>
    %1 = tosa.bitwise_not %0 : (tensor<94x32x95x98x1x12xi1>) -> tensor<94x32x95x98x1x12xi1>
    %2 = tosa.equal %arg2, %arg3 : (tensor<55x10xi32>, tensor<1x1xi32>) -> tensor<55x10xi1>
    return %1, %2 : tensor<94x32x95x98x1x12xi1>, tensor<55x10xi1>
  }
}
