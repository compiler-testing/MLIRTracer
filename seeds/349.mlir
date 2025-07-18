module {
  func.func @main(%arg0: tensor<99x53x92x15x81xi1>, %arg1: tensor<1x53x1x1x1xi1>, %arg2: tensor<16x90x62x55xi1>, %arg3: tensor<f32>) -> (tensor<99x53x92x15x81xi1>, tensor<1x90x62x55xi1>, tensor<f32>) {
    %0 = tosa.logical_and %arg0, %arg1 : (tensor<99x53x92x15x81xi1>, tensor<1x53x1x1x1xi1>) -> tensor<99x53x92x15x81xi1>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<99x53x92x15x81xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<99x53x92x15x81xi1>
    %2 = tosa.abs %1 : (tensor<99x53x92x15x81xi1>) -> tensor<99x53x92x15x81xi1>
    %3 = tosa.reduce_all %arg2 {axis = 0 : i32} : (tensor<16x90x62x55xi1>) -> tensor<1x90x62x55xi1>
    %4 = tosa.exp %arg3 : (tensor<f32>) -> tensor<f32>
    return %2, %3, %4 : tensor<99x53x92x15x81xi1>, tensor<1x90x62x55xi1>, tensor<f32>
  }
}
