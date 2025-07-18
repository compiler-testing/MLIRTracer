module {
  func.func @main(%arg0: tensor<80x81x99xi1>, %arg1: tensor<1x81x99xi1>, %arg2: tensor<41xf32>) -> (tensor<80x81x99xi1>, tensor<41xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<80x81x99xi1>, tensor<1x81x99xi1>) -> tensor<80x81x99xi1>
    %1 = tosa.log %arg2 : (tensor<41xf32>) -> tensor<41xf32>
    %2 = tosa.reciprocal %1 : (tensor<41xf32>) -> tensor<41xf32>
    return %0, %2 : tensor<80x81x99xi1>, tensor<41xf32>
  }
}
