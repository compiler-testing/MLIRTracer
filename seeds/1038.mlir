module {
  func.func @main(%arg0: tensor<18x51x98x4xi64>) -> tensor<54x153x294x4xi64> {
    %t_0 = tosa.const_shape {values = dense<[ 3, 3, 3, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<18x51x98x4xi64>, !tosa.shape<4>) -> tensor<54x153x294x4xi64>
    %1 = tosa.bitwise_and %0, %0 : (tensor<54x153x294x4xi64>, tensor<54x153x294x4xi64>) -> tensor<54x153x294x4xi64>
    return %1 : tensor<54x153x294x4xi64>
  }
}
