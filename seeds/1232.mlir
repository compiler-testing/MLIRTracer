module {
  func.func @main(%arg0: tensor<47xi64>, %arg1: tensor<24x82x59x11x77x28xf32>) -> (tensor<24x82x59x11x77x28xf32>, tensor<1x1x1x1xi64>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<47xi64>) -> tensor<47xi64>
    %1 = tosa.clamp %0 {min_val = -28 : i64, max_val = 77 : i64} : (tensor<47xi64>) -> tensor<47xi64>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<47xi64>) -> tensor<1xi64>
    %3 = tosa.exp %arg1 : (tensor<24x82x59x11x77x28xf32>) -> tensor<24x82x59x11x77x28xf32>
    %r_4 = tosa.const_shape {values = dense<[ 1, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.reshape %2, %r_4 : (tensor<1xi64>, !tosa.shape<4>) -> tensor<1x1x1x1xi64>
    %5 = tosa.reverse %4 {axis = 1 : i32} : (tensor<1x1x1x1xi64>) -> tensor<1x1x1x1xi64>
    return %3, %5 : tensor<24x82x59x11x77x28xf32>, tensor<1x1x1x1xi64>
  }
}
