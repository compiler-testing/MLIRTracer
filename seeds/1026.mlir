module {
  func.func @main(%arg0: tensor<99xi1>, %arg1: tensor<99xi1>, %arg2: tensor<11x13x92x83xf32>) -> (tensor<11x13x92x83xf32>, tensor<1xi1>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<99xi1>, tensor<99xi1>) -> tensor<99xi1>
    %1 = tosa.logical_right_shift %0, %0 : (tensor<99xi1>, tensor<99xi1>) -> tensor<99xi1>
    %2 = tosa.ceil %arg2 : (tensor<11x13x92x83xf32>) -> tensor<11x13x92x83xf32>
    %3 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<99xi1>) -> tensor<1xi1>
    return %2, %3 : tensor<11x13x92x83xf32>, tensor<1xi1>
  }
}
