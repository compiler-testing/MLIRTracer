module {
  func.func @main(%arg0: tensor<95x98x75x92x43x19xi64>, %arg1: tensor<1x1x75x92x1x19xi64>, %arg2: tensor<31x19xf32>, %arg3: tensor<89x87x76x48x99xi1>, %arg4: tensor<34x81x27x90xi1>) -> (tensor<95x98x75x92x43x19xi64>, tensor<2x3xf32>, tensor<89x87x76x48x99xi1>, tensor<34x81x1x90xi1>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<95x98x75x92x43x19xi64>, tensor<1x1x75x92x1x19xi64>) -> tensor<95x98x75x92x43x19xi64>
    %1 = tosa.tanh %arg2 : (tensor<31x19xf32>) -> tensor<31x19xf32>
    %s_2_start = tosa.const_shape {values = dense<[ 19, 16 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_2_size = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<31x19xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x3xf32>
    %3 = tosa.logical_not %arg3 : (tensor<89x87x76x48x99xi1>) -> tensor<89x87x76x48x99xi1>
    %4 = tosa.reduce_all %arg4 {axis = 2 : i32} : (tensor<34x81x27x90xi1>) -> tensor<34x81x1x90xi1>
    return %0, %2, %3, %4 : tensor<95x98x75x92x43x19xi64>, tensor<2x3xf32>, tensor<89x87x76x48x99xi1>, tensor<34x81x1x90xi1>
  }
}
