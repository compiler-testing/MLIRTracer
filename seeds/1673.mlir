module {
  func.func @main(%arg0: tensor<59x59x28xi8>, %arg1: tensor<97x52x99x36xf32>) -> (tensor<59x59x1xi1>, tensor<59x59x1xi1>, tensor<1x11x5xi1>, tensor<97x52x99x36xf32>) {
    %0 = tosa.reduce_max %arg0 {axis = 2 : i32} : (tensor<59x59x28xi8>) -> tensor<59x59x1xi8>
    %1 = tosa.equal %0, %0 : (tensor<59x59x1xi8>, tensor<59x59x1xi8>) -> tensor<59x59x1xi1>
    %2 = tosa.add %1, %1 : (tensor<59x59x1xi1>, tensor<59x59x1xi1>) -> tensor<59x59x1xi1>
    %3 = tosa.minimum %0, %0 : (tensor<59x59x1xi8>, tensor<59x59x1xi8>) -> tensor<59x59x1xi8>
    %4 = tosa.greater_equal %0, %3 : (tensor<59x59x1xi8>, tensor<59x59x1xi8>) -> tensor<59x59x1xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 33, 48, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_5_size = tosa.const_shape {values = dense<[ 4, 11, 5 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.slice %2, %s_5_start, %s_5_size : (tensor<59x59x1xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x11x5xi1>
    %6 = tosa.bitwise_xor %5, %5 : (tensor<4x11x5xi1>, tensor<4x11x5xi1>) -> tensor<4x11x5xi1>
    %7 = tosa.greater_equal %0, %0 : (tensor<59x59x1xi8>, tensor<59x59x1xi8>) -> tensor<59x59x1xi1>
    %8 = tosa.reduce_all %6 {axis = 0 : i32} : (tensor<4x11x5xi1>) -> tensor<1x11x5xi1>
    %9 = tosa.reciprocal %arg1 : (tensor<97x52x99x36xf32>) -> tensor<97x52x99x36xf32>
    return %4, %7, %8, %9 : tensor<59x59x1xi1>, tensor<59x59x1xi1>, tensor<1x11x5xi1>, tensor<97x52x99x36xf32>
  }
}
