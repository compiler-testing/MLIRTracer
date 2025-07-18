module {
  func.func @main(%arg0: tensor<55x4x79x3xi64>, %arg1: tensor<11x20xf32>) -> (tensor<55x2x237x3xi64>, tensor<11x20xf32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 3 : i32} : (tensor<55x4x79x3xi64>) -> tensor<55x4x79x1xi64>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<55x4x79x1xi64>) -> tensor<55x1x79x1xi64>
    %t_2 = tosa.const_shape {values = dense<[ 1, 2, 3, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.tile %1, %t_2 : (tensor<55x1x79x1xi64>, !tosa.shape<4>) -> tensor<55x2x237x3xi64>
    %3 = tosa.log %arg1 : (tensor<11x20xf32>) -> tensor<11x20xf32>
    return %2, %3 : tensor<55x2x237x3xi64>, tensor<11x20xf32>
  }
}
