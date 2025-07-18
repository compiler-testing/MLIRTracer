module {
  func.func @main(%arg0: tensor<99x62xi16>, %arg1: tensor<99x1xi16>, %arg2: tensor<83x21x72x23x57xi32>, %arg3: tensor<1x1x72x23x1xi32>, %arg4: tensor<28x72x46x67x48x91xf32>) -> (tensor<83x21x72x23x57xi1>, tensor<1x62xi16>, tensor<28x72x46x67x48x91xf32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<99x62xi16>, tensor<99x1xi16>) -> tensor<99x62xi16>
    %1 = tosa.logical_left_shift %0, %0 : (tensor<99x62xi16>, tensor<99x62xi16>) -> tensor<99x62xi16>
    %2 = tosa.add %1, %1 : (tensor<99x62xi16>, tensor<99x62xi16>) -> tensor<99x62xi16>
    %3 = tosa.greater_equal %arg2, %arg3 : (tensor<83x21x72x23x57xi32>, tensor<1x1x72x23x1xi32>) -> tensor<83x21x72x23x57xi1>
    %4 = tosa.reduce_min %2 {axis = 0 : i32} : (tensor<99x62xi16>) -> tensor<1x62xi16>
    %5 = tosa.ceil %arg4 : (tensor<28x72x46x67x48x91xf32>) -> tensor<28x72x46x67x48x91xf32>
    %6 = tosa.bitwise_and %3, %3 : (tensor<83x21x72x23x57xi1>, tensor<83x21x72x23x57xi1>) -> tensor<83x21x72x23x57xi1>
    %7 = tosa.arithmetic_right_shift %6, %6 {round = false} : (tensor<83x21x72x23x57xi1>, tensor<83x21x72x23x57xi1>) -> tensor<83x21x72x23x57xi1>
    %8 = tosa.sub %4, %4 : (tensor<1x62xi16>, tensor<1x62xi16>) -> tensor<1x62xi16>
    %9 = tosa.log %5 : (tensor<28x72x46x67x48x91xf32>) -> tensor<28x72x46x67x48x91xf32>
    return %7, %8, %9 : tensor<83x21x72x23x57xi1>, tensor<1x62xi16>, tensor<28x72x46x67x48x91xf32>
  }
}
