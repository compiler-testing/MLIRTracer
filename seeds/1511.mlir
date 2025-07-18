module {
  func.func @main(%arg0: tensor<99x22x26xi1>) -> tensor<99x22x1xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 2 : i32} : (tensor<99x22x26xi1>) -> tensor<99x22x1xi1>
    return %0 : tensor<99x22x1xi1>
  }
}
