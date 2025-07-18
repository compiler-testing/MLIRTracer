module {
  func.func @main(%arg0: tensor<53x98x64xi1>, %arg1: tensor<53x98x64xi1>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<4x7xi64>, %arg5: tensor<4x1xi64>) -> (tensor<f32>, tensor<53x294x128xi1>, tensor<4x7xi1>, tensor<9x4xi64>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<53x98x64xi1>, tensor<53x98x64xi1>) -> tensor<53x98x64xi1>
    %t_1 = tosa.const_shape {values = dense<[ 1, 3, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.tile %0, %t_1 : (tensor<53x98x64xi1>, !tosa.shape<3>) -> tensor<53x294x128xi1>
    %2 = tosa.bitwise_or %1, %1 : (tensor<53x294x128xi1>, tensor<53x294x128xi1>) -> tensor<53x294x128xi1>
    %3 = tosa.pow %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %in_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_4 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<f32>, tensor<1xf32>, tensor<1xf32>) -> tensor<f32>
    %5 = tosa.minimum %arg4, %arg5 : (tensor<4x7xi64>, tensor<4x1xi64>) -> tensor<4x7xi64>
    %6 = tosa.logical_not %2 : (tensor<53x294x128xi1>) -> tensor<53x294x128xi1>
    %s_7_start = tosa.const_shape {values = dense<[ 0, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_7_size = tosa.const_shape {values = dense<[ 9, 4 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %7 = tosa.slice %5, %s_7_start, %s_7_size : (tensor<4x7xi64>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<9x4xi64>
    %8 = tosa.tanh %4 : (tensor<f32>) -> tensor<f32>
    %t_9 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.tile %6, %t_9 : (tensor<53x294x128xi1>, !tosa.shape<3>) -> tensor<53x294x128xi1>
    %10 = tosa.greater %5, %5 : (tensor<4x7xi64>, tensor<4x7xi64>) -> tensor<4x7xi1>
    %11 = tosa.logical_right_shift %7, %7 : (tensor<9x4xi64>, tensor<9x4xi64>) -> tensor<9x4xi64>
    return %8, %9, %10, %11 : tensor<f32>, tensor<53x294x128xi1>, tensor<4x7xi1>, tensor<9x4xi64>
  }
}
