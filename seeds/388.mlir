module {
  func.func @main(%arg0: tensor<15xi8>, %arg1: tensor<63x67xi1>, %arg2: tensor<1x1xi1>, %arg3: tensor<4xf32>) -> (tensor<6xi8>, tensor<4xf32>, tensor<63x67xi1>) {
    %0 = tosa.abs %arg0 : (tensor<15xi8>) -> tensor<15xi8>
    %1 = tosa.logical_xor %arg1, %arg2 : (tensor<63x67xi1>, tensor<1x1xi1>) -> tensor<63x67xi1>
    %2 = tosa.bitwise_not %1 : (tensor<63x67xi1>) -> tensor<63x67xi1>
    %s_3_start = tosa.const_shape {values = dense<[ 5 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_3_size = tosa.const_shape {values = dense<[ 6 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.slice %0, %s_3_start, %s_3_size : (tensor<15xi8>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<6xi8>
    %4 = tosa.tanh %arg3 : (tensor<4xf32>) -> tensor<4xf32>
    %5 = tosa.exp %4 : (tensor<4xf32>) -> tensor<4xf32>
    %6 = tosa.logical_and %2, %2 : (tensor<63x67xi1>, tensor<63x67xi1>) -> tensor<63x67xi1>
    return %3, %5, %6 : tensor<6xi8>, tensor<4xf32>, tensor<63x67xi1>
  }
}
