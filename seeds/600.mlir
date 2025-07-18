module {
  func.func @main(%arg0: tensor<74x4x26x39x31xi1>, %arg1: tensor<99x68x73x77xi1>, %arg2: tensor<97x59x64x30x13x79xf32>) -> (tensor<74x4x26x39x31xi1>, tensor<99x68x73x77xi1>, tensor<97x59x64x30x13x79xf32>) {
    %0 = tosa.identity %arg0 : (tensor<74x4x26x39x31xi1>) -> tensor<74x4x26x39x31xi1>
    %1 = tosa.reverse %arg1 {axis = 1 : i32} : (tensor<99x68x73x77xi1>) -> tensor<99x68x73x77xi1>
    %2 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<99x68x73x77xi1>, tensor<99x68x73x77xi1>) -> tensor<99x68x73x77xi1>
    %3 = tosa.reciprocal %arg2 : (tensor<97x59x64x30x13x79xf32>) -> tensor<97x59x64x30x13x79xf32>
    %4 = tosa.reciprocal %3 : (tensor<97x59x64x30x13x79xf32>) -> tensor<97x59x64x30x13x79xf32>
    return %0, %2, %4 : tensor<74x4x26x39x31xi1>, tensor<99x68x73x77xi1>, tensor<97x59x64x30x13x79xf32>
  }
}
