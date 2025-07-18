module {
  func.func @main(%arg0: tensor<28x72x35x85xi64>) -> tensor<28x216x70x170xi64> {
    %t_0 = tosa.const_shape {values = dense<[ 1, 3, 2, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<28x72x35x85xi64>, !tosa.shape<4>) -> tensor<28x216x70x170xi64>
    return %0 : tensor<28x216x70x170xi64>
  }
}
