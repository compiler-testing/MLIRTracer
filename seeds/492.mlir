module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<84xf32>, %arg2: tensor<100x11x61xi64>, %arg3: tensor<100x1x1xi64>) -> (tensor<84xf32>, tensor<100x11x61xi64>, tensor<f32>) {
    %0 = tosa.exp %arg0 : (tensor<f32>) -> tensor<f32>
    %t_1 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.tile %arg1, %t_1 : (tensor<84xf32>, !tosa.shape<1>) -> tensor<84xf32>
    %2 = tosa.logical_right_shift %arg2, %arg3 : (tensor<100x11x61xi64>, tensor<100x1x1xi64>) -> tensor<100x11x61xi64>
    %3 = tosa.tanh %0 : (tensor<f32>) -> tensor<f32>
    return %1, %2, %3 : tensor<84xf32>, tensor<100x11x61xi64>, tensor<f32>
  }
}
