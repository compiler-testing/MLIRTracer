module {
  func.func @main(%arg0: tensor<8x37x70x31x17xi1>, %arg1: tensor<43x73x94x67x95x64xf32>, %arg2: tensor<1x73x1x67x1x64xf32>, %arg3: tensor<47x36x84xf32>) -> (tensor<8x37x70x31x17xi1>, tensor<43x73x94x67x95x64xi1>, tensor<47x36x84xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<8x37x70x31x17xi1>) -> tensor<8x37x70x31x17xi1>
    %1 = tosa.logical_not %0 : (tensor<8x37x70x31x17xi1>) -> tensor<8x37x70x31x17xi1>
    %2 = tosa.greater_equal %arg1, %arg2 : (tensor<43x73x94x67x95x64xf32>, tensor<1x73x1x67x1x64xf32>) -> tensor<43x73x94x67x95x64xi1>
    %3 = tosa.reciprocal %arg3 : (tensor<47x36x84xf32>) -> tensor<47x36x84xf32>
    return %1, %2, %3 : tensor<8x37x70x31x17xi1>, tensor<43x73x94x67x95x64xi1>, tensor<47x36x84xf32>
  }
}
