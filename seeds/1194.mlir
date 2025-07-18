module {
  func.func @main(%arg0: tensor<54x46x47x82x54xi8>, %arg1: tensor<1x46x47x1x54xi8>, %arg2: tensor<25xi1>, %arg3: tensor<25xi1>, %arg4: tensor<61x88x91x91x24xi32>, %arg5: tensor<61x1x91x1x24xi32>) -> (tensor<54x46x47x82x54xi8>, tensor<25xi1>, tensor<61x88x91x91x24xi32>) {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<54x46x47x82x54xi8>, tensor<1x46x47x1x54xi8>) -> tensor<54x46x47x82x54xi8>
    %1 = tosa.logical_or %arg2, %arg3 : (tensor<25xi1>, tensor<25xi1>) -> tensor<25xi1>
    %2 = tosa.intdiv %arg4, %arg5 : (tensor<61x88x91x91x24xi32>, tensor<61x1x91x1x24xi32>) -> tensor<61x88x91x91x24xi32>
    return %0, %1, %2 : tensor<54x46x47x82x54xi8>, tensor<25xi1>, tensor<61x88x91x91x24xi32>
  }
}
