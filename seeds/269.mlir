module {
  func.func @main(%arg0: tensor<23x74x26x68xi1>, %arg1: tensor<23x74x1x68xi1>, %arg2: tensor<f32>, %arg3: tensor<97x20xi64>, %arg4: tensor<1x20xi64>) -> (tensor<1x74x26x68xi1>, tensor<2x60xi64>, tensor<f32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<23x74x26x68xi1>, tensor<23x74x1x68xi1>) -> tensor<23x74x26x68xi1>
    %1 = tosa.floor %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.maximum %arg3, %arg4 : (tensor<97x20xi64>, tensor<1x20xi64>) -> tensor<97x20xi64>
    %3 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<97x20xi64>) -> tensor<1x20xi64>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4 = tosa.negate %3, %in_zp_4, %out_zp_4 : (tensor<1x20xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x20xi64>
    %5 = tosa.reduce_all %0 {axis = 0 : i32} : (tensor<23x74x26x68xi1>) -> tensor<1x74x26x68xi1>
    %6 = tosa.arithmetic_right_shift %5, %5 {round = false} : (tensor<1x74x26x68xi1>, tensor<1x74x26x68xi1>) -> tensor<1x74x26x68xi1>
    %t_7 = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %7 = tosa.tile %4, %t_7 : (tensor<1x20xi64>, !tosa.shape<2>) -> tensor<2x60xi64>
    %8 = tosa.ceil %1 : (tensor<f32>) -> tensor<f32>
    return %6, %7, %8 : tensor<1x74x26x68xi1>, tensor<2x60xi64>, tensor<f32>
  }
}
