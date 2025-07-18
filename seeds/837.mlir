module {
  func.func @main(%arg0: tensor<89x23x86xi1>, %arg1: tensor<1x1x86xi1>, %arg2: tensor<75x3xf32>) -> (tensor<75x3xf32>, tensor<89x23x1xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<89x23x86xi1>, tensor<1x1x86xi1>) -> tensor<89x23x86xi1>
    %1 = tosa.logical_and %0, %0 : (tensor<89x23x86xi1>, tensor<89x23x86xi1>) -> tensor<89x23x86xi1>
    %2 = tosa.tanh %arg2 : (tensor<75x3xf32>) -> tensor<75x3xf32>
    %3 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<89x23x86xi1>) -> tensor<89x23x1xi1>
    return %2, %3 : tensor<75x3xf32>, tensor<89x23x1xi1>
  }
}
