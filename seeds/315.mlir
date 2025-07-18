module {
  func.func @main(%arg0: tensor<22xi64>) -> tensor<66xi64> {
    %t_0 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<22xi64>, !tosa.shape<1>) -> tensor<66xi64>
    return %0 : tensor<66xi64>
  }
}
