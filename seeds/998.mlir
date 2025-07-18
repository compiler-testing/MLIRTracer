module {
  func.func @main(%arg0: tensor<91x66xi32>, %arg1: tensor<99x75xf32>, %arg2: tensor<1x1xf32>) -> (tensor<1x1x1x1xi32>, tensor<99x75xf32>, tensor<1x1xi32>, tensor<297x150xf32>) {
    %s_0_start = tosa.const_shape {values = dense<[ 78, 6 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_0_size = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.slice %arg0, %s_0_start, %s_0_size : (tensor<91x66xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x1xi32>
    %r_1 = tosa.const_shape {values = dense<[ 1, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.reshape %0, %r_1 : (tensor<1x1xi32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
    %2 = tosa.pow %arg1, %arg2 : (tensor<99x75xf32>, tensor<1x1xf32>) -> tensor<99x75xf32>
    %3 = tosa.rsqrt %2 : (tensor<99x75xf32>) -> tensor<99x75xf32>
    %4 = tosa.abs %3 : (tensor<99x75xf32>) -> tensor<99x75xf32>
    %5 = tosa.reverse %2 {axis = 1 : i32} : (tensor<99x75xf32>) -> tensor<99x75xf32>
    %6 = tosa.clamp %4 {min_val = 6.300000e+01 : f32, max_val = 1.220000e+02 : f32} : (tensor<99x75xf32>) -> tensor<99x75xf32>
    %7 = tosa.reverse %6 {axis = 1 : i32} : (tensor<99x75xf32>) -> tensor<99x75xf32>
    %8 = tosa.pow %7, %7 : (tensor<99x75xf32>, tensor<99x75xf32>) -> tensor<99x75xf32>
    %t_9 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %9 = tosa.tile %8, %t_9 : (tensor<99x75xf32>, !tosa.shape<2>) -> tensor<297x150xf32>
    %10 = tosa.clz %0 : (tensor<1x1xi32>) -> tensor<1x1xi32>
    %11 = tosa.exp %9 : (tensor<297x150xf32>) -> tensor<297x150xf32>
    return %1, %5, %10, %11 : tensor<1x1x1x1xi32>, tensor<99x75xf32>, tensor<1x1xi32>, tensor<297x150xf32>
  }
}
