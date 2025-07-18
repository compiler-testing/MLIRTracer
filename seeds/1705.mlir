module {
  func.func @main(%arg0: tensor<26xi64>, %arg1: tensor<55x60x91x52xi1>, %arg2: tensor<23x67x56x60x5xf32>) -> (tensor<1x1x13x2xi64>, tensor<55x60x91x52xi1>, tensor<23x67x56x60x5xf32>) {
    %r_0 = tosa.const_shape {values = dense<[ 1, 1, 13, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<26xi64>, !tosa.shape<4>) -> tensor<1x1x13x2xi64>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<1x1x13x2xi64>, tensor<1x1x13x2xi64>) -> tensor<1x1x13x2xi64>
    %2 = tosa.logical_not %arg1 : (tensor<55x60x91x52xi1>) -> tensor<55x60x91x52xi1>
    %3 = tosa.exp %arg2 : (tensor<23x67x56x60x5xf32>) -> tensor<23x67x56x60x5xf32>
    return %1, %2, %3 : tensor<1x1x13x2xi64>, tensor<55x60x91x52xi1>, tensor<23x67x56x60x5xf32>
  }
}
