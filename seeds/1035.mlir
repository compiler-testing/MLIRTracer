module {
  func.func @main(%arg0: tensor<19x16x81xi64>, %arg1: tensor<i1>, %arg2: tensor<i1>, %arg3: tensor<95x77x76xi32>, %arg4: tensor<1x1x1xi32>, %arg5: tensor<f32>) -> (tensor<38x1x243xi64>, tensor<i1>, tensor<95x77x76xi32>, tensor<f32>) {
    %t_0 = tosa.const_shape {values = dense<[ 2, 1, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<19x16x81xi64>, !tosa.shape<3>) -> tensor<38x16x243xi64>
    %1 = tosa.minimum %0, %0 : (tensor<38x16x243xi64>, tensor<38x16x243xi64>) -> tensor<38x16x243xi64>
    %2 = tosa.reduce_max %1 {axis = 1 : i32} : (tensor<38x16x243xi64>) -> tensor<38x1x243xi64>
    %3 = tosa.logical_and %arg1, %arg2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.intdiv %arg3, %arg4 : (tensor<95x77x76xi32>, tensor<1x1x1xi32>) -> tensor<95x77x76xi32>
    %5 = tosa.log %arg5 : (tensor<f32>) -> tensor<f32>
    return %2, %3, %4, %5 : tensor<38x1x243xi64>, tensor<i1>, tensor<95x77x76xi32>, tensor<f32>
  }
}
