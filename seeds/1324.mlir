module {
  func.func @main(%arg0: tensor<24x87x49x43xi64>) -> tensor<48x261x98x86xi64> {
    %t_0 = tosa.const_shape {values = dense<[ 2, 3, 2, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<24x87x49x43xi64>, !tosa.shape<4>) -> tensor<48x261x98x86xi64>
    return %0 : tensor<48x261x98x86xi64>
  }
}
