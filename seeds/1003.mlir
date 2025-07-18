module {
  func.func @main(%arg0: tensor<20x99xf32>) -> (tensor<20x99xf32>, tensor<20x99xf32>, tensor<20x99xi1>) {
    %0 = tosa.rsqrt %arg0 : (tensor<20x99xf32>) -> tensor<20x99xf32>
    %1 = tosa.greater %0, %0 : (tensor<20x99xf32>, tensor<20x99xf32>) -> tensor<20x99xi1>
    %2 = tosa.tanh %0 : (tensor<20x99xf32>) -> tensor<20x99xf32>
    %3 = tosa.bitwise_and %1, %1 : (tensor<20x99xi1>, tensor<20x99xi1>) -> tensor<20x99xi1>
    %4 = tosa.rsqrt %0 : (tensor<20x99xf32>) -> tensor<20x99xf32>
    %5 = tosa.identity %3 : (tensor<20x99xi1>) -> tensor<20x99xi1>
    %6 = tosa.logical_left_shift %5, %3 : (tensor<20x99xi1>, tensor<20x99xi1>) -> tensor<20x99xi1>
    return %2, %4, %6 : tensor<20x99xf32>, tensor<20x99xf32>, tensor<20x99xi1>
  }
}
