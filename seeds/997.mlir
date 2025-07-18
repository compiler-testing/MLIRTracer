module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<99x67x4x25xi1>) -> (tensor<99x1x4x25xi1>, tensor<i1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.logical_not %arg1 : (tensor<99x67x4x25xi1>) -> tensor<99x67x4x25xi1>
    %2 = tosa.reduce_min %1 {axis = 1 : i32} : (tensor<99x67x4x25xi1>) -> tensor<99x1x4x25xi1>
    %3 = tosa.bitwise_xor %2, %2 : (tensor<99x1x4x25xi1>, tensor<99x1x4x25xi1>) -> tensor<99x1x4x25xi1>
    %4 = tosa.equal %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %3, %4 : tensor<99x1x4x25xi1>, tensor<i1>
  }
}
