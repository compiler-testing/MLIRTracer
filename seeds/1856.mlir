module {
  func.func @main(%arg0: tensor<99x63xf32>, %arg1: tensor<12xi64>, %arg2: tensor<1xi64>, %arg3: tensor<89x64x28x1xi1>) -> (tensor<12xi64>, tensor<99x63xf32>, tensor<89x1x28x1xi1>, tensor<1xi64>) {
    %0 = tosa.ceil %arg0 : (tensor<99x63xf32>) -> tensor<99x63xf32>
    %1 = tosa.logical_right_shift %arg1, %arg2 : (tensor<12xi64>, tensor<1xi64>) -> tensor<12xi64>
    %2 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<12xi64>, tensor<12xi64>) -> tensor<12xi64>
    %3 = tosa.identity %2 : (tensor<12xi64>) -> tensor<12xi64>
    %4 = tosa.add %3, %3 : (tensor<12xi64>, tensor<12xi64>) -> tensor<12xi64>
    %5 = tosa.tanh %0 : (tensor<99x63xf32>) -> tensor<99x63xf32>
    %6 = tosa.reduce_any %arg3 {axis = 1 : i32} : (tensor<89x64x28x1xi1>) -> tensor<89x1x28x1xi1>
    %7 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<12xi64>) -> tensor<1xi64>
    return %4, %5, %6, %7 : tensor<12xi64>, tensor<99x63xf32>, tensor<89x1x28x1xi1>, tensor<1xi64>
  }
}
