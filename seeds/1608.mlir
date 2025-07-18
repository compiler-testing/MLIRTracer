module {
  func.func @main(%arg0: tensor<6x54x6xi64>, %arg1: tensor<39x32x9x71x7xf32>, %arg2: tensor<1x1x1x1x7xf32>) -> (tensor<6x54x6xi64>, tensor<39x32x9x71x7xf32>) {
    %0 = tosa.abs %arg0 : (tensor<6x54x6xi64>) -> tensor<6x54x6xi64>
    %t_1 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %0, %t_1 : (tensor<6x54x6xi64>, !tosa.shape<3>) -> tensor<6x54x6xi64>
    %2 = tosa.pow %arg1, %arg2 : (tensor<39x32x9x71x7xf32>, tensor<1x1x1x1x7xf32>) -> tensor<39x32x9x71x7xf32>
    return %1, %2 : tensor<6x54x6xi64>, tensor<39x32x9x71x7xf32>
  }
}
