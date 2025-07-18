module {
  func.func @main(%arg0: tensor<61xf32>, %arg1: tensor<37x100x75x90xi32>, %arg2: tensor<37x100x1x90xi32>) -> (tensor<61xf32>, tensor<37x100x75x90xi32>) {
    %0 = tosa.sigmoid %arg0 : (tensor<61xf32>) -> tensor<61xf32>
    %1 = tosa.maximum %0, %0 : (tensor<61xf32>, tensor<61xf32>) -> tensor<61xf32>
    %2 = tosa.intdiv %arg1, %arg2 : (tensor<37x100x75x90xi32>, tensor<37x100x1x90xi32>) -> tensor<37x100x75x90xi32>
    return %1, %2 : tensor<61xf32>, tensor<37x100x75x90xi32>
  }
}
