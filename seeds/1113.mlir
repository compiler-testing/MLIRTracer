module {
  func.func @main(%arg0: tensor<46xi32>, %arg1: tensor<46xi32>, %arg2: tensor<54x22x91x33x89x1xf32>, %arg3: tensor<31x30x7x2xi1>, %arg4: tensor<1x1x1x2xi1>) -> (tensor<46xi32>, tensor<54x22x91x33x89x1xf32>, tensor<31x30x7x2xi1>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<46xi32>, tensor<46xi32>) -> tensor<46xi32>
    %1 = tosa.sub %0, %0 : (tensor<46xi32>, tensor<46xi32>) -> tensor<46xi32>
    %2 = tosa.floor %arg2 : (tensor<54x22x91x33x89x1xf32>) -> tensor<54x22x91x33x89x1xf32>
    %3 = tosa.floor %2 : (tensor<54x22x91x33x89x1xf32>) -> tensor<54x22x91x33x89x1xf32>
    %4 = tosa.logical_xor %arg3, %arg4 : (tensor<31x30x7x2xi1>, tensor<1x1x1x2xi1>) -> tensor<31x30x7x2xi1>
    %5 = tosa.pow %3, %2 : (tensor<54x22x91x33x89x1xf32>, tensor<54x22x91x33x89x1xf32>) -> tensor<54x22x91x33x89x1xf32>
    %6 = tosa.pow %5, %2 : (tensor<54x22x91x33x89x1xf32>, tensor<54x22x91x33x89x1xf32>) -> tensor<54x22x91x33x89x1xf32>
    %7 = tosa.maximum %2, %6 : (tensor<54x22x91x33x89x1xf32>, tensor<54x22x91x33x89x1xf32>) -> tensor<54x22x91x33x89x1xf32>
    %8 = tosa.bitwise_or %4, %4 : (tensor<31x30x7x2xi1>, tensor<31x30x7x2xi1>) -> tensor<31x30x7x2xi1>
    return %1, %7, %8 : tensor<46xi32>, tensor<54x22x91x33x89x1xf32>, tensor<31x30x7x2xi1>
  }
}
