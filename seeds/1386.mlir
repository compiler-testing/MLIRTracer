module {
  func.func @main(%arg0: tensor<99x72x97x18x34x91xf32>, %arg1: tensor<70xi32>) -> (tensor<99x72x97x18x34x91xf32>, tensor<2xi32>) {
    %0 = tosa.exp %arg0 : (tensor<99x72x97x18x34x91xf32>) -> tensor<99x72x97x18x34x91xf32>
    %1 = tosa.reduce_max %arg1 {axis = 0 : i32} : (tensor<70xi32>) -> tensor<1xi32>
    %2 = tosa.concat %1, %1 {axis = 0 : i32} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %3 = tosa.abs %2 : (tensor<2xi32>) -> tensor<2xi32>
    %4 = tosa.clz %3 : (tensor<2xi32>) -> tensor<2xi32>
    return %0, %4 : tensor<99x72x97x18x34x91xf32>, tensor<2xi32>
  }
}
