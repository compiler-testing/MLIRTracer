module {
  func.func @main(%arg0: tensor<51x76x70x53x34xi1>, %arg1: tensor<1x1x1x1x1xi1>, %arg2: tensor<46x52x41x11xi32>, %arg3: tensor<46x1x1x11xi32>) -> (tensor<51x76x70x53x34xi1>, tensor<46x52x41x11xi1>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<51x76x70x53x34xi1>, tensor<1x1x1x1x1xi1>) -> tensor<51x76x70x53x34xi1>
    %1 = tosa.greater_equal %arg2, %arg3 : (tensor<46x52x41x11xi32>, tensor<46x1x1x11xi32>) -> tensor<46x52x41x11xi1>
    return %0, %1 : tensor<51x76x70x53x34xi1>, tensor<46x52x41x11xi1>
  }
}
