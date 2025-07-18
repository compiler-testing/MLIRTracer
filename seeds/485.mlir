module {
  func.func @main(%arg0: tensor<75x59x45x63xi16>) -> tensor<1x5x1x55755xi32> {
    %0 = tosa.argmax %arg0 {axis = 2 : i32} : (tensor<75x59x45x63xi16>) -> tensor<75x59x63xi32>
    %r_1 = tosa.const_shape {values = dense<[ 1, 5, 1, 55755 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.reshape %0, %r_1 : (tensor<75x59x63xi32>, !tosa.shape<4>) -> tensor<1x5x1x55755xi32>
    %2 = tosa.bitwise_xor %1, %1 : (tensor<1x5x1x55755xi32>, tensor<1x5x1x55755xi32>) -> tensor<1x5x1x55755xi32>
    %3 = tosa.logical_left_shift %2, %2 : (tensor<1x5x1x55755xi32>, tensor<1x5x1x55755xi32>) -> tensor<1x5x1x55755xi32>
    return %3 : tensor<1x5x1x55755xi32>
  }
}
