module {
  func.func @main(%arg0: tensor<5x6xi1>, %arg1: tensor<4x48x37x19xi8>, %arg2: tensor<4x1x1x1xi8>) -> (tensor<4x48x37x19xi8>, tensor<5x6xi1>) {
    %0 = tosa.logical_not %arg0 : (tensor<5x6xi1>) -> tensor<5x6xi1>
    %1 = tosa.maximum %arg1, %arg2 : (tensor<4x48x37x19xi8>, tensor<4x1x1x1xi8>) -> tensor<4x48x37x19xi8>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<5x6xi1>, tensor<5x6xi1>) -> tensor<5x6xi1>
    %3 = tosa.logical_not %2 : (tensor<5x6xi1>) -> tensor<5x6xi1>
    return %1, %3 : tensor<4x48x37x19xi8>, tensor<5x6xi1>
  }
}
