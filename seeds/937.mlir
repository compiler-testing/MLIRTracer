module {
  func.func @main(%arg0: tensor<54x36xi8>) -> tensor<1x36xi1> {
    %t_0 = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<54x36xi8>, !tosa.shape<2>) -> tensor<108x36xi8>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<108x36xi8>, tensor<108x36xi8>) -> tensor<108x36xi8>
    %2 = tosa.equal %1, %0 : (tensor<108x36xi8>, tensor<108x36xi8>) -> tensor<108x36xi1>
    %3 = tosa.bitwise_not %2 : (tensor<108x36xi1>) -> tensor<108x36xi1>
    %4 = tosa.logical_or %3, %3 : (tensor<108x36xi1>, tensor<108x36xi1>) -> tensor<108x36xi1>
    %5 = tosa.reduce_any %4 {axis = 0 : i32} : (tensor<108x36xi1>) -> tensor<1x36xi1>
    %6 = tosa.bitwise_xor %5, %5 : (tensor<1x36xi1>, tensor<1x36xi1>) -> tensor<1x36xi1>
    %7 = tosa.logical_and %6, %5 : (tensor<1x36xi1>, tensor<1x36xi1>) -> tensor<1x36xi1>
    return %7 : tensor<1x36xi1>
  }
}
