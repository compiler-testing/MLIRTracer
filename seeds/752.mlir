module {
  func.func @main(%arg0: tensor<75x67x53x75x58x13xi1>, %arg1: tensor<1x1x1x1x58x1xi1>, %arg2: tensor<4x73x50x85x44x89xf32>) -> (tensor<75x67x53x75x58x13xi1>, tensor<4x73x50x85x44x89xf32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<75x67x53x75x58x13xi1>, tensor<1x1x1x1x58x1xi1>) -> tensor<75x67x53x75x58x13xi1>
    %1 = tosa.sigmoid %arg2 : (tensor<4x73x50x85x44x89xf32>) -> tensor<4x73x50x85x44x89xf32>
    return %0, %1 : tensor<75x67x53x75x58x13xi1>, tensor<4x73x50x85x44x89xf32>
  }
}
