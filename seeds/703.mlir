module {
  func.func @main(%arg0: tensor<99x69xi16>) -> tensor<1x69xi16> {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<99x69xi16>) -> tensor<1x69xi16>
    return %0 : tensor<1x69xi16>
  }
}
