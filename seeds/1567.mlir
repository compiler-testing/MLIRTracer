module {
  func.func @main(%arg0: tensor<99x33x67x75xf32>, %arg1: tensor<1x1x67x75xf32>, %arg2: tensor<23x64x47xi1>, %arg3: tensor<1x64x1xi1>) -> (tensor<198x33x201x150xf32>, tensor<99x33x67x75xf32>, tensor<23x1x47xi1>, tensor<99x33x67x75xf32>, tensor<5x4x8xi1>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<99x33x67x75xf32>, tensor<1x1x67x75xf32>) -> tensor<99x33x67x75xf32>
    %t_1 = tosa.const_shape {values = dense<[ 2, 1, 3, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.tile %0, %t_1 : (tensor<99x33x67x75xf32>, !tosa.shape<4>) -> tensor<198x33x201x150xf32>
    %2 = tosa.arithmetic_right_shift %arg2, %arg3 {round = false} : (tensor<23x64x47xi1>, tensor<1x64x1xi1>) -> tensor<23x64x47xi1>
    %3 = tosa.logical_or %2, %2 : (tensor<23x64x47xi1>, tensor<23x64x47xi1>) -> tensor<23x64x47xi1>
    %4 = tosa.sub %2, %2 : (tensor<23x64x47xi1>, tensor<23x64x47xi1>) -> tensor<23x64x47xi1>
    %5 = tosa.bitwise_and %4, %4 : (tensor<23x64x47xi1>, tensor<23x64x47xi1>) -> tensor<23x64x47xi1>
    %6 = tosa.log %0 : (tensor<99x33x67x75xf32>) -> tensor<99x33x67x75xf32>
    %7 = tosa.reduce_sum %3 {axis = 1 : i32} : (tensor<23x64x47xi1>) -> tensor<23x1x47xi1>
    %8 = tosa.sub %5, %3 : (tensor<23x64x47xi1>, tensor<23x64x47xi1>) -> tensor<23x64x47xi1>
    %9 = tosa.floor %0 : (tensor<99x33x67x75xf32>) -> tensor<99x33x67x75xf32>
    %s_10_start = tosa.const_shape {values = dense<[ 18, 19, 18 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_10_size = tosa.const_shape {values = dense<[ 5, 4, 8 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %10 = tosa.slice %8, %s_10_start, %s_10_size : (tensor<23x64x47xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<5x4x8xi1>
    return %1, %6, %7, %9, %10 : tensor<198x33x201x150xf32>, tensor<99x33x67x75xf32>, tensor<23x1x47xi1>, tensor<99x33x67x75xf32>, tensor<5x4x8xi1>
  }
}
