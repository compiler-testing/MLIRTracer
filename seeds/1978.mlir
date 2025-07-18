module {
  func.func @main(%arg0: tensor<99x36x71x74x49x73xf32>, %arg1: tensor<64x62x68x2x17x98xi1>, %arg2: tensor<1x62x1x1x1x98xi1>) -> (tensor<64x62x68x2x17x98xi1>, tensor<99x36x71x74x49x73xf32>) {
    %0 = tosa.exp %arg0 : (tensor<99x36x71x74x49x73xf32>) -> tensor<99x36x71x74x49x73xf32>
    %1 = tosa.bitwise_and %arg1, %arg2 : (tensor<64x62x68x2x17x98xi1>, tensor<1x62x1x1x1x98xi1>) -> tensor<64x62x68x2x17x98xi1>
    %2 = tosa.identity %1 : (tensor<64x62x68x2x17x98xi1>) -> tensor<64x62x68x2x17x98xi1>
    %3 = tosa.ceil %0 : (tensor<99x36x71x74x49x73xf32>) -> tensor<99x36x71x74x49x73xf32>
    return %2, %3 : tensor<64x62x68x2x17x98xi1>, tensor<99x36x71x74x49x73xf32>
  }
}
