module {
  func.func @main(%arg0: tensor<63x37x72xi1>, %arg1: tensor<33x28x1x30xf32>) -> (tensor<189x74x216xi1>, tensor<33x28x1x30xf32>, tensor<63x37x72xi1>, tensor<33x30x28x1xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<63x37x72xi1>) -> tensor<63x37x72xi1>
    %t_1 = tosa.const_shape {values = dense<[ 3, 2, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %0, %t_1 : (tensor<63x37x72xi1>, !tosa.shape<3>) -> tensor<189x74x216xi1>
    %2 = tosa.reciprocal %arg1 : (tensor<33x28x1x30xf32>) -> tensor<33x28x1x30xf32>
    %3 = tosa.minimum %2, %2 : (tensor<33x28x1x30xf32>, tensor<33x28x1x30xf32>) -> tensor<33x28x1x30xf32>
    %4 = tosa.bitwise_or %0, %0 : (tensor<63x37x72xi1>, tensor<63x37x72xi1>) -> tensor<63x37x72xi1>
    %5 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %6 = tosa.transpose %2 {perms = array<i32: 0, 3, 1, 2>} : (tensor<33x28x1x30xf32>) -> tensor<33x30x28x1xf32>
    %7 = tosa.greater %6, %6 : (tensor<33x30x28x1xf32>, tensor<33x30x28x1xf32>) -> tensor<33x30x28x1xi1>
    return %1, %3, %4, %7 : tensor<189x74x216xi1>, tensor<33x28x1x30xf32>, tensor<63x37x72xi1>, tensor<33x30x28x1xi1>
  }
}
