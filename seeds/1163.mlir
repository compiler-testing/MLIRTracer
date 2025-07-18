module {
  func.func @main(%arg0: tensor<60xi8>) -> tensor<1xi8> {
    %s_0_start = tosa.const_shape {values = dense<[ 46 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_0_size = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<60xi8>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<1xi8>
    return %0 : tensor<1xi8>
  }
}
