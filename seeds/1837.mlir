module {
  func.func @main(%arg0: tensor<1xi32>, %arg1: tensor<47xi1>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<1xi1>, tensor<6xi32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %1 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<47xi1>) -> tensor<1xi1>
    %2 = tosa.exp %arg2 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.logical_or %1, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.logical_right_shift %3, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %5 = tosa.bitwise_not %1 : (tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.reciprocal %2 : (tensor<f32>) -> tensor<f32>
    %7 = tosa.ceil %2 : (tensor<f32>) -> tensor<f32>
    %8 = tosa.tanh %6 : (tensor<f32>) -> tensor<f32>
    %9 = tosa.logical_xor %4, %5 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %s_10_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_10_size = tosa.const_shape {values = dense<[ 6 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %10 = tosa.slice %0, %s_10_start, %s_10_size : (tensor<1xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<6xi32>
    return %7, %8, %9, %10 : tensor<f32>, tensor<f32>, tensor<1xi1>, tensor<6xi32>
  }
}
