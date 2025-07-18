module {
  func.func @main(%arg0: tensor<80x32xi8>) -> tensor<1x32xi8> {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<80x32xi8>) -> tensor<1x32xi8>
    %r_1 = tosa.const_shape {values = dense<[ 1, 32 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<1x32xi8>, !tosa.shape<2>) -> tensor<1x32xi8>
    return %1 : tensor<1x32xi8>
  }
}
