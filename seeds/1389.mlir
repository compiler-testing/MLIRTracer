module {
  func.func @main(%arg0: tensor<15x52xi64>, %arg1: tensor<8x77x86x63xi1>, %arg2: tensor<99x79x45x79x24xf32>, %arg3: tensor<1x79x45x79x24xf32>) -> (tensor<15x52xi64>, tensor<1x77x86x63xi1>, tensor<1x77x86x1xi1>, tensor<99x79x45x79x24xf32>) {
    %0 = tosa.clz %arg0 : (tensor<15x52xi64>) -> tensor<15x52xi64>
    %1 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<8x77x86x63xi1>) -> tensor<1x77x86x63xi1>
    %2 = tosa.arithmetic_right_shift %1, %1 {round = false} : (tensor<1x77x86x63xi1>, tensor<1x77x86x63xi1>) -> tensor<1x77x86x63xi1>
    %3 = tosa.reduce_any %2 {axis = 3 : i32} : (tensor<1x77x86x63xi1>) -> tensor<1x77x86x1xi1>
    %4 = tosa.logical_xor %2, %2 : (tensor<1x77x86x63xi1>, tensor<1x77x86x63xi1>) -> tensor<1x77x86x63xi1>
    %5 = tosa.pow %arg2, %arg3 : (tensor<99x79x45x79x24xf32>, tensor<1x79x45x79x24xf32>) -> tensor<99x79x45x79x24xf32>
    %6 = tosa.reduce_product %3 {axis = 0 : i32} : (tensor<1x77x86x1xi1>) -> tensor<1x77x86x1xi1>
    %7 = tosa.minimum %5, %5 : (tensor<99x79x45x79x24xf32>, tensor<99x79x45x79x24xf32>) -> tensor<99x79x45x79x24xf32>
    return %0, %4, %6, %7 : tensor<15x52xi64>, tensor<1x77x86x63xi1>, tensor<1x77x86x1xi1>, tensor<99x79x45x79x24xf32>
  }
}
