module {
  func.func @main(%arg0: tensor<2x99xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<47xi1>, %arg3: tensor<1xi1>) -> (tensor<47xi1>, tensor<10x8xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<2x99xf32>, !tosa.shape<4>, tensor<1xf32>) -> tensor<2x99xf32>
    %s_1_start = tosa.const_shape {values = dense<[ 0, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_1_size = tosa.const_shape {values = dense<[ 10, 8 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<2x99xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<10x8xf32>
    %2 = tosa.logical_xor %arg2, %arg3 : (tensor<47xi1>, tensor<1xi1>) -> tensor<47xi1>
    %3 = tosa.tanh %1 : (tensor<10x8xf32>) -> tensor<10x8xf32>
    return %2, %3 : tensor<47xi1>, tensor<10x8xf32>
  }
}
