module {
  func.func @main(%arg0: tensor<99x97xi1>, %arg1: tensor<99x1xi1>) -> tensor<1x97xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<99x97xi1>, tensor<99x1xi1>) -> tensor<99x97xi1>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<99x97xi1>, tensor<99x97xi1>) -> tensor<99x97xi1>
    %2 = tosa.bitwise_or %1, %1 : (tensor<99x97xi1>, tensor<99x97xi1>) -> tensor<99x97xi1>
    %3 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<99x97xi1>) -> tensor<1x97xi1>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<1x97xi1>, tensor<1x97xi1>) -> tensor<1x97xi1>
    %5 = tosa.clz %4 : (tensor<1x97xi1>) -> tensor<1x97xi1>
    return %5 : tensor<1x97xi1>
  }
}
