module {
  func.func @main(%arg0: tensor<79xi8>, %arg1: tensor<20xi8>, %arg2: tensor<16x81x34x46xi1>) -> (tensor<99xi8>, tensor<i32>, tensor<16x1x34x46xi1>, tensor<1x1xi32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<79xi8>, tensor<20xi8>) -> tensor<99xi8>
    %1 = tosa.argmax %0 {axis = 0 : i32} : (tensor<99xi8>) -> tensor<i32>
    %2 = tosa.reverse %0 {axis = 0 : i32} : (tensor<99xi8>) -> tensor<99xi8>
    %3 = tosa.argmax %0 {axis = 0 : i32} : (tensor<99xi8>) -> tensor<i32>
    %4 = tosa.reduce_any %arg2 {axis = 1 : i32} : (tensor<16x81x34x46xi1>) -> tensor<16x1x34x46xi1>
    %r_5 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %1, %r_5 : (tensor<i32>, !tosa.shape<2>) -> tensor<1x1xi32>
    return %2, %3, %4, %5 : tensor<99xi8>, tensor<i32>, tensor<16x1x34x46xi1>, tensor<1x1xi32>
  }
}
