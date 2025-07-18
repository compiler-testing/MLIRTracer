module {
  func.func @main(%arg0: tensor<66x45x9x5x17xi1>, %arg1: tensor<66x1x1x5x17xi1>, %arg2: tensor<99x56x38x23xi32>, %arg3: tensor<67x51x49x37x39x19xf32>, %arg4: tensor<42xi1>) -> (tensor<66x45x9x5x17xi1>, tensor<99x56x1x23xi32>, tensor<67x51x49x37x39x19xf32>, tensor<1xi1>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<66x45x9x5x17xi1>, tensor<66x1x1x5x17xi1>) -> tensor<66x45x9x5x17xi1>
    %1 = tosa.logical_not %0 : (tensor<66x45x9x5x17xi1>) -> tensor<66x45x9x5x17xi1>
    %2 = tosa.clz %1 : (tensor<66x45x9x5x17xi1>) -> tensor<66x45x9x5x17xi1>
    %3 = tosa.reduce_sum %arg2 {axis = 2 : i32} : (tensor<99x56x38x23xi32>) -> tensor<99x56x1x23xi32>
    %4 = tosa.reverse %3 {axis = 2 : i32} : (tensor<99x56x1x23xi32>) -> tensor<99x56x1x23xi32>
    %5 = tosa.sigmoid %arg3 : (tensor<67x51x49x37x39x19xf32>) -> tensor<67x51x49x37x39x19xf32>
    %6 = tosa.logical_not %2 : (tensor<66x45x9x5x17xi1>) -> tensor<66x45x9x5x17xi1>
    %7 = tosa.reciprocal %5 : (tensor<67x51x49x37x39x19xf32>) -> tensor<67x51x49x37x39x19xf32>
    %8 = tosa.maximum %3, %4 : (tensor<99x56x1x23xi32>, tensor<99x56x1x23xi32>) -> tensor<99x56x1x23xi32>
    %9 = tosa.sigmoid %7 : (tensor<67x51x49x37x39x19xf32>) -> tensor<67x51x49x37x39x19xf32>
    %10 = tosa.reduce_any %arg4 {axis = 0 : i32} : (tensor<42xi1>) -> tensor<1xi1>
    %11 = tosa.exp %9 : (tensor<67x51x49x37x39x19xf32>) -> tensor<67x51x49x37x39x19xf32>
    %12 = tosa.logical_left_shift %10, %10 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %6, %8, %11, %12 : tensor<66x45x9x5x17xi1>, tensor<99x56x1x23xi32>, tensor<67x51x49x37x39x19xf32>, tensor<1xi1>
  }
}
