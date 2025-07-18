module {
  func.func @main(%arg0: tensor<27xi16>, %arg1: tensor<27xi16>, %arg2: tensor<63x68xf32>) -> (tensor<1x2xi1>, tensor<63x68xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<27xi16>, tensor<27xi16>) -> tensor<27xi16>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<27xi16>) -> tensor<i32>
    %2 = tosa.greater %1, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %r_3 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.reshape %2, %r_3 : (tensor<i1>, !tosa.shape<2>) -> tensor<1x1xi1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %5 = tosa.reduce_all %4 {axis = 1 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %t_6 = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.tile %5, %t_6 : (tensor<1x1xi1>, !tosa.shape<2>) -> tensor<1x2xi1>
    %7 = tosa.exp %arg2 : (tensor<63x68xf32>) -> tensor<63x68xf32>
    return %6, %7 : tensor<1x2xi1>, tensor<63x68xf32>
  }
}
