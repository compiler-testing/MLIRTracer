module {
  func.func @main(%arg0: tensor<10x53x30x58xi64>) -> tensor<1590x1xi64> {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<10x53x30x58xi64>) -> tensor<10x53x30x58xi64>
    %r_1 = tosa.const_shape {values = dense<[ 795, 1160 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<10x53x30x58xi64>, !tosa.shape<2>) -> tensor<795x1160xi64>
    %2 = tosa.reduce_product %1 {axis = 1 : i32} : (tensor<795x1160xi64>) -> tensor<795x1xi64>
    %3 = tosa.concat %2, %2 {axis = 0 : i32} : (tensor<795x1xi64>, tensor<795x1xi64>) -> tensor<1590x1xi64>
    return %3 : tensor<1590x1xi64>
  }
}
