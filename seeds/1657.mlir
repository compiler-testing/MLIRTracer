module {
  func.func @main(%arg0: tensor<97x9x10xi1>, %arg1: tensor<97x10x51xi1>, %arg2: tensor<23x40x69xi32>, %arg3: tensor<23x1x1xi32>, %arg4: tensor<9x4x40xf32>) -> (tensor<97x9x51xi1>, tensor<23x80x69xi1>, tensor<23x40x69xi32>, tensor<23x40x69xi32>, tensor<9x4x40xf32>, tensor<9x4x40xi1>, tensor<23x40x69xi32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<97x9x10xi1>, tensor<97x10x51xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<97x9x51xi1>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<23x40x69xi32>, tensor<23x1x1xi32>) -> tensor<23x40x69xi32>
    %2 = tosa.bitwise_not %0 : (tensor<97x9x51xi1>) -> tensor<97x9x51xi1>
    %3 = tosa.greater_equal %1, %1 : (tensor<23x40x69xi32>, tensor<23x40x69xi32>) -> tensor<23x40x69xi1>
    %4 = tosa.logical_or %3, %3 : (tensor<23x40x69xi1>, tensor<23x40x69xi1>) -> tensor<23x40x69xi1>
    %5 = tosa.log %arg4 : (tensor<9x4x40xf32>) -> tensor<9x4x40xf32>
    %6 = tosa.concat %4, %4 {axis = 1 : i32} : (tensor<23x40x69xi1>, tensor<23x40x69xi1>) -> tensor<23x80x69xi1>
    %7 = tosa.clz %6 : (tensor<23x80x69xi1>) -> tensor<23x80x69xi1>
    %8 = tosa.greater_equal %5, %5 : (tensor<9x4x40xf32>, tensor<9x4x40xf32>) -> tensor<9x4x40xi1>
    %9 = tosa.sub %1, %1 : (tensor<23x40x69xi32>, tensor<23x40x69xi32>) -> tensor<23x40x69xi32>
    %10 = tosa.logical_left_shift %9, %9 : (tensor<23x40x69xi32>, tensor<23x40x69xi32>) -> tensor<23x40x69xi32>
    %11 = tosa.bitwise_and %10, %1 : (tensor<23x40x69xi32>, tensor<23x40x69xi32>) -> tensor<23x40x69xi32>
    %12 = tosa.bitwise_not %11 : (tensor<23x40x69xi32>) -> tensor<23x40x69xi32>
    %13 = tosa.logical_xor %8, %8 : (tensor<9x4x40xi1>, tensor<9x4x40xi1>) -> tensor<9x4x40xi1>
    %14 = tosa.bitwise_and %12, %12 : (tensor<23x40x69xi32>, tensor<23x40x69xi32>) -> tensor<23x40x69xi32>
    %15 = tosa.sub %12, %1 : (tensor<23x40x69xi32>, tensor<23x40x69xi32>) -> tensor<23x40x69xi32>
    %16 = tosa.floor %5 : (tensor<9x4x40xf32>) -> tensor<9x4x40xf32>
    %17 = tosa.logical_or %13, %13 : (tensor<9x4x40xi1>, tensor<9x4x40xi1>) -> tensor<9x4x40xi1>
    %18 = tosa.minimum %12, %11 : (tensor<23x40x69xi32>, tensor<23x40x69xi32>) -> tensor<23x40x69xi32>
    return %2, %7, %14, %15, %16, %17, %18 : tensor<97x9x51xi1>, tensor<23x80x69xi1>, tensor<23x40x69xi32>, tensor<23x40x69xi32>, tensor<9x4x40xf32>, tensor<9x4x40xi1>, tensor<23x40x69xi32>
  }
}
