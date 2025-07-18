module {
  func.func @main(%arg0: tensor<61x99xi16>) -> tensor<99xi32> {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<61x99xi16>) -> tensor<99xi32>
    return %0 : tensor<99xi32>
  }
}
