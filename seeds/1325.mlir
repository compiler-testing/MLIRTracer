module {
  func.func @main(%arg0: tensor<99x20xi8>, %arg1: tensor<99x19xi8>, %arg2: tensor<21x32x60xi1>, %arg3: tensor<1x1x1xi1>) -> (tensor<21x32x60xi1>, tensor<99x39xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<99x20xi8>, tensor<99x19xi8>) -> tensor<99x39xi8>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<99x39xi8>, tensor<99x39xi8>) -> tensor<99x39xi8>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<21x32x60xi1>, tensor<1x1x1xi1>) -> tensor<21x32x60xi1>
    %3 = tosa.greater %1, %0 : (tensor<99x39xi8>, tensor<99x39xi8>) -> tensor<99x39xi1>
    return %2, %3 : tensor<21x32x60xi1>, tensor<99x39xi1>
  }
}
