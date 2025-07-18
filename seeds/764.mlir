module {
  func.func @main(%arg0: tensor<59x37x98x60xi8>, %arg1: tensor<69xf32>) -> (tensor<69xf32>, tensor<2x3x7x9xi8>) {
    %0 = tosa.abs %arg0 : (tensor<59x37x98x60xi8>) -> tensor<59x37x98x60xi8>
    %1 = tosa.reverse %0 {axis = 1 : i32} : (tensor<59x37x98x60xi8>) -> tensor<59x37x98x60xi8>
    %2 = tosa.tanh %arg1 : (tensor<69xf32>) -> tensor<69xf32>
    %3 = tosa.add %1, %0 : (tensor<59x37x98x60xi8>, tensor<59x37x98x60xi8>) -> tensor<59x37x98x60xi8>
    %s_4_start = tosa.const_shape {values = dense<[ 7, 34, 50, 24 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_4_size = tosa.const_shape {values = dense<[ 2, 3, 7, 9 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.slice %3, %s_4_start, %s_4_size : (tensor<59x37x98x60xi8>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x3x7x9xi8>
    %5 = tosa.bitwise_not %4 : (tensor<2x3x7x9xi8>) -> tensor<2x3x7x9xi8>
    return %2, %5 : tensor<69xf32>, tensor<2x3x7x9xi8>
  }
}
