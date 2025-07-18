module {
  func.func @main(%arg0: tensor<12xf32>, %arg1: tensor<99x91xi1>, %arg2: tensor<1x91xi1>) -> (tensor<6x2x1xf32>, tensor<1x12xi1>, tensor<99x1xi1>) {
    %0 = tosa.tanh %arg0 : (tensor<12xf32>) -> tensor<12xf32>
    %r_1 = tosa.const_shape {values = dense<[ 1, 12 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.reshape %0, %r_1 : (tensor<12xf32>, !tosa.shape<2>) -> tensor<1x12xf32>
    %2 = tosa.rsqrt %1 : (tensor<1x12xf32>) -> tensor<1x12xf32>
    %3 = tosa.logical_xor %arg1, %arg2 : (tensor<99x91xi1>, tensor<1x91xi1>) -> tensor<99x91xi1>
    %r_4 = tosa.const_shape {values = dense<[ 6, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.reshape %2, %r_4 : (tensor<1x12xf32>, !tosa.shape<3>) -> tensor<6x2x1xf32>
    %5 = tosa.bitwise_or %3, %3 : (tensor<99x91xi1>, tensor<99x91xi1>) -> tensor<99x91xi1>
    %6 = tosa.floor %1 : (tensor<1x12xf32>) -> tensor<1x12xf32>
    %7 = tosa.reduce_max %5 {axis = 1 : i32} : (tensor<99x91xi1>) -> tensor<99x1xi1>
    %8 = tosa.sub %7, %7 : (tensor<99x1xi1>, tensor<99x1xi1>) -> tensor<99x1xi1>
    %9 = tosa.greater_equal %6, %2 : (tensor<1x12xf32>, tensor<1x12xf32>) -> tensor<1x12xi1>
    %10 = tosa.reverse %8 {axis = 1 : i32} : (tensor<99x1xi1>) -> tensor<99x1xi1>
    return %4, %9, %10 : tensor<6x2x1xf32>, tensor<1x12xi1>, tensor<99x1xi1>
  }
}
