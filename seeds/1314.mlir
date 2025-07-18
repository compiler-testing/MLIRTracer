module {
  func.func @main(%arg0: tensor<18x3x43x71xi64>) -> tensor<54x6x43x213xi64> {
    %t_0 = tosa.const_shape {values = dense<[ 3, 2, 1, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<18x3x43x71xi64>, !tosa.shape<4>) -> tensor<54x6x43x213xi64>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<54x6x43x213xi64>, tensor<54x6x43x213xi64>) -> tensor<54x6x43x213xi64>
    return %1 : tensor<54x6x43x213xi64>
  }
}
