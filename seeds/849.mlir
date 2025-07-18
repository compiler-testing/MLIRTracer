module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<99x87xi8>) -> (tensor<f32>, tensor<99x174xi8>) {
    %0 = tosa.tanh %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.rsqrt %0 : (tensor<f32>) -> tensor<f32>
    %t_2 = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.tile %arg1, %t_2 : (tensor<99x87xi8>, !tosa.shape<2>) -> tensor<99x174xi8>
    %3 = tosa.reverse %2 {axis = 1 : i32} : (tensor<99x174xi8>) -> tensor<99x174xi8>
    %4 = tosa.add %2, %3 : (tensor<99x174xi8>, tensor<99x174xi8>) -> tensor<99x174xi8>
    return %1, %4 : tensor<f32>, tensor<99x174xi8>
  }
}
