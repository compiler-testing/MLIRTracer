module {
  func.func @main(%arg0: tensor<51x37xf32>, %arg1: tensor<35xi1>, %arg2: tensor<35xi1>) -> (tensor<35xi1>, tensor<51x37xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<51x37xf32>) -> tensor<51x37xf32>
    %1 = tosa.reciprocal %0 : (tensor<51x37xf32>) -> tensor<51x37xf32>
    %2 = tosa.sub %1, %1 : (tensor<51x37xf32>, tensor<51x37xf32>) -> tensor<51x37xf32>
    %3 = tosa.bitwise_or %arg1, %arg2 : (tensor<35xi1>, tensor<35xi1>) -> tensor<35xi1>
    %4 = tosa.greater %2, %1 : (tensor<51x37xf32>, tensor<51x37xf32>) -> tensor<51x37xi1>
    return %3, %4 : tensor<35xi1>, tensor<51x37xi1>
  }
}
