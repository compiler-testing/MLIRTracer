module {
  func.func @main(%arg0: tensor<6x11xi64>) -> tensor<6x22xi64> {
    %0 = tosa.identity %arg0 : (tensor<6x11xi64>) -> tensor<6x11xi64>
    %t_1 = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %0, %t_1 : (tensor<6x11xi64>, !tosa.shape<2>) -> tensor<6x22xi64>
    return %1 : tensor<6x22xi64>
  }
}
