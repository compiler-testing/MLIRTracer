module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<63x3x47xi1>) -> (tensor<i8>, tensor<10x7x9xi1>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %s_1_start = tosa.const_shape {values = dense<[ 53, 0, 22 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_1_size = tosa.const_shape {values = dense<[ 10, 7, 9 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.slice %arg2, %s_1_start, %s_1_size : (tensor<63x3x47xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<10x7x9xi1>
    return %0, %1 : tensor<i8>, tensor<10x7x9xi1>
  }
}
