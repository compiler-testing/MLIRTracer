module {
  func.func @main(%arg0: tensor<5xi1>, %arg1: tensor<1xi1>, %arg2: tensor<50x40xi32>, %arg3: tensor<50x1xi32>, %arg4: tensor<8x28x47x50xf32>, %arg5: tensor<1x1x1x1xf32>) -> (tensor<50x40xi32>, tensor<8x28x47x50xf32>, tensor<5xi1>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<5xi1>, tensor<1xi1>) -> tensor<5xi1>
    %1 = tosa.maximum %arg2, %arg3 : (tensor<50x40xi32>, tensor<50x1xi32>) -> tensor<50x40xi32>
    %2 = tosa.pow %arg4, %arg5 : (tensor<8x28x47x50xf32>, tensor<1x1x1x1xf32>) -> tensor<8x28x47x50xf32>
    %3 = tosa.reciprocal %2 : (tensor<8x28x47x50xf32>) -> tensor<8x28x47x50xf32>
    %4 = tosa.logical_and %0, %0 : (tensor<5xi1>, tensor<5xi1>) -> tensor<5xi1>
    return %1, %3, %4 : tensor<50x40xi32>, tensor<8x28x47x50xf32>, tensor<5xi1>
  }
}
