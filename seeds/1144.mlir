module {
  func.func @main(%arg0: tensor<44x53xf32>, %arg1: tensor<99x53xf32>) -> tensor<143x53xf32> {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<44x53xf32>, tensor<99x53xf32>) -> tensor<143x53xf32>
    %1 = tosa.tanh %0 : (tensor<143x53xf32>) -> tensor<143x53xf32>
    return %1 : tensor<143x53xf32>
  }
}
