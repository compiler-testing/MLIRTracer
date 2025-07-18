module {
  func.func @main(%arg0: tensor<24x1x27x43xi64>) -> tensor<72x1x27x1xi64> {
    %t_0 = tosa.const_shape {values = dense<[ 3, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<24x1x27x43xi64>, !tosa.shape<4>) -> tensor<72x1x27x43xi64>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<72x1x27x43xi64>, tensor<72x1x27x43xi64>) -> tensor<72x1x27x43xi64>
    %2 = tosa.reduce_sum %1 {axis = 3 : i32} : (tensor<72x1x27x43xi64>) -> tensor<72x1x27x1xi64>
    %3 = tosa.identity %2 : (tensor<72x1x27x1xi64>) -> tensor<72x1x27x1xi64>
    return %3 : tensor<72x1x27x1xi64>
  }
}
