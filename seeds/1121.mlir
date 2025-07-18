module {
  func.func @main(%arg0: tensor<56x13xi16>, %arg1: tensor<56x13xi16>, %arg2: tensor<47x90xi1>, %arg3: tensor<73x7xi32>, %arg4: tensor<1x7xi32>, %arg5: tensor<13x4x17xf32>) -> (tensor<56x13xi16>, tensor<73x7xi32>, tensor<1x90xi1>, tensor<13x4x17xi1>, tensor<1x1xi1>, tensor<13x4x17xf32>, tensor<1x90xi1>, tensor<13x4x17xf32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<56x13xi16>, tensor<56x13xi16>) -> tensor<56x13xi16>
    %1 = tosa.clz %0 : (tensor<56x13xi16>) -> tensor<56x13xi16>
    %2 = tosa.logical_not %arg2 : (tensor<47x90xi1>) -> tensor<47x90xi1>
    %3 = tosa.bitwise_and %2, %2 : (tensor<47x90xi1>, tensor<47x90xi1>) -> tensor<47x90xi1>
    %4 = tosa.sub %3, %2 : (tensor<47x90xi1>, tensor<47x90xi1>) -> tensor<47x90xi1>
    %5 = tosa.intdiv %arg3, %arg4 : (tensor<73x7xi32>, tensor<1x7xi32>) -> tensor<73x7xi32>
    %6 = tosa.reduce_product %4 {axis = 0 : i32} : (tensor<47x90xi1>) -> tensor<1x90xi1>
    %7 = tosa.reduce_product %6 {axis = 0 : i32} : (tensor<1x90xi1>) -> tensor<1x90xi1>
    %8 = tosa.rsqrt %arg5 : (tensor<13x4x17xf32>) -> tensor<13x4x17xf32>
    %9 = tosa.greater_equal %8, %8 : (tensor<13x4x17xf32>, tensor<13x4x17xf32>) -> tensor<13x4x17xi1>
    %10 = tosa.arithmetic_right_shift %9, %9 {round = true} : (tensor<13x4x17xi1>, tensor<13x4x17xi1>) -> tensor<13x4x17xi1>
    %11 = tosa.maximum %8, %8 : (tensor<13x4x17xf32>, tensor<13x4x17xf32>) -> tensor<13x4x17xf32>
    %12 = tosa.reduce_sum %6 {axis = 1 : i32} : (tensor<1x90xi1>) -> tensor<1x1xi1>
    %13 = tosa.minimum %11, %8 : (tensor<13x4x17xf32>, tensor<13x4x17xf32>) -> tensor<13x4x17xf32>
    %14 = tosa.logical_and %6, %6 : (tensor<1x90xi1>, tensor<1x90xi1>) -> tensor<1x90xi1>
    %15 = tosa.abs %8 : (tensor<13x4x17xf32>) -> tensor<13x4x17xf32>
    return %1, %5, %7, %10, %12, %13, %14, %15 : tensor<56x13xi16>, tensor<73x7xi32>, tensor<1x90xi1>, tensor<13x4x17xi1>, tensor<1x1xi1>, tensor<13x4x17xf32>, tensor<1x90xi1>, tensor<13x4x17xf32>
  }
}
