module {
  func.func @main(%arg0: tensor<19x26x80x96xi16>, %arg1: tensor<4x2xi32>, %arg2: tensor<60x28x63x46x65xi32>, %arg3: tensor<60x28x1x46x1xi32>) -> (tensor<19x26x80x1xi16>, tensor<60x28x63x46x65xi32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<8xindex>} : () -> !tosa.shape<8>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<19x26x80x96xi16>, !tosa.shape<8>, tensor<1xi16>) -> tensor<19x26x80x96xi16>
    %1 = tosa.reduce_sum %0 {axis = 3 : i32} : (tensor<19x26x80x96xi16>) -> tensor<19x26x80x1xi16>
    %2 = tosa.intdiv %arg2, %arg3 : (tensor<60x28x63x46x65xi32>, tensor<60x28x1x46x1xi32>) -> tensor<60x28x63x46x65xi32>
    return %1, %2 : tensor<19x26x80x1xi16>, tensor<60x28x63x46x65xi32>
  }
}
