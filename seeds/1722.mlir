module {
  func.func @main(%arg0: tensor<44x47x10xf32>, %arg1: tensor<82x36xi32>, %arg2: tensor<82x36xi32>) -> (tensor<82x36xi32>, tensor<44x47x10xi1>) {
    %0 = tosa.tanh %arg0 : (tensor<44x47x10xf32>) -> tensor<44x47x10xf32>
    %1 = tosa.intdiv %arg1, %arg2 : (tensor<82x36xi32>, tensor<82x36xi32>) -> tensor<82x36xi32>
    %2 = tosa.greater_equal %0, %0 : (tensor<44x47x10xf32>, tensor<44x47x10xf32>) -> tensor<44x47x10xi1>
    return %1, %2 : tensor<82x36xi32>, tensor<44x47x10xi1>
  }
}
