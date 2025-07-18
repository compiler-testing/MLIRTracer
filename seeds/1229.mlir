module {
  func.func @main(%arg0: tensor<48xi32>, %arg1: tensor<1xi32>, %arg2: tensor<32x95x92x46x8x66xi1>, %arg3: tensor<1x1x92x1x8x1xi1>) -> (tensor<48xi32>, tensor<32x95x92x46x8x66xi1>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<48xi32>, tensor<1xi32>) -> tensor<48xi32>
    %1 = tosa.logical_xor %arg2, %arg3 : (tensor<32x95x92x46x8x66xi1>, tensor<1x1x92x1x8x1xi1>) -> tensor<32x95x92x46x8x66xi1>
    %2 = tosa.bitwise_and %0, %0 : (tensor<48xi32>, tensor<48xi32>) -> tensor<48xi32>
    %3 = tosa.logical_xor %1, %1 : (tensor<32x95x92x46x8x66xi1>, tensor<32x95x92x46x8x66xi1>) -> tensor<32x95x92x46x8x66xi1>
    return %2, %3 : tensor<48xi32>, tensor<32x95x92x46x8x66xi1>
  }
}
