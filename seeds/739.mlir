module {
  func.func @main(%arg0: tensor<33x44x20x15xi8>, %arg1: tensor<33x44x20x1xi8>, %arg2: tensor<12x35xf32>, %arg3: tensor<12x1xf32>, %arg4: tensor<71xf32>) -> (tensor<12x35xi1>, tensor<33x44x20x15xi1>, tensor<71xf32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<33x44x20x15xi8>, tensor<33x44x20x1xi8>) -> tensor<33x44x20x15xi1>
    %1 = tosa.equal %arg2, %arg3 : (tensor<12x35xf32>, tensor<12x1xf32>) -> tensor<12x35xi1>
    %2 = tosa.ceil %arg4 : (tensor<71xf32>) -> tensor<71xf32>
    %3 = tosa.bitwise_or %0, %0 : (tensor<33x44x20x15xi1>, tensor<33x44x20x15xi1>) -> tensor<33x44x20x15xi1>
    %4 = tosa.tanh %2 : (tensor<71xf32>) -> tensor<71xf32>
    return %1, %3, %4 : tensor<12x35xi1>, tensor<33x44x20x15xi1>, tensor<71xf32>
  }
}
