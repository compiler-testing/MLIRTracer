module {
  func.func @main(%arg0: tensor<95x9x39xi16>, %arg1: tensor<95x39x45xi16>, %arg2: tensor<41x6x20x59xi1>, %arg3: tensor<41x6x1x1xi1>, %arg4: tensor<99x46x97xf32>) -> (tensor<95x9x45xi16>, tensor<41x6x20x59xi1>, tensor<99x46x97xi1>, tensor<41x6x20x1xi1>, tensor<41x6x20x1xi1>, tensor<99x1x97xf32>, tensor<41x6x20x1xi1>, tensor<99x1x97xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<95x9x39xi16>, tensor<95x39x45xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<95x9x45xi16>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<95x9x45xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<95x9x45xi16>
    %2 = tosa.abs %1 : (tensor<95x9x45xi16>) -> tensor<95x9x45xi16>
    %3 = tosa.arithmetic_right_shift %2, %0 {round = false} : (tensor<95x9x45xi16>, tensor<95x9x45xi16>) -> tensor<95x9x45xi16>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<95x9x45xi16>, tensor<95x9x45xi16>) -> tensor<95x9x45xi16>
    %5 = tosa.logical_and %arg2, %arg3 : (tensor<41x6x20x59xi1>, tensor<41x6x1x1xi1>) -> tensor<41x6x20x59xi1>
    %6 = tosa.sigmoid %arg4 : (tensor<99x46x97xf32>) -> tensor<99x46x97xf32>
    %7 = tosa.minimum %6, %6 : (tensor<99x46x97xf32>, tensor<99x46x97xf32>) -> tensor<99x46x97xf32>
    %8 = tosa.bitwise_or %5, %5 : (tensor<41x6x20x59xi1>, tensor<41x6x20x59xi1>) -> tensor<41x6x20x59xi1>
    %9 = tosa.maximum %6, %7 : (tensor<99x46x97xf32>, tensor<99x46x97xf32>) -> tensor<99x46x97xf32>
    %10 = tosa.greater %9, %6 : (tensor<99x46x97xf32>, tensor<99x46x97xf32>) -> tensor<99x46x97xi1>
    %11 = tosa.log %6 : (tensor<99x46x97xf32>) -> tensor<99x46x97xf32>
    %12 = tosa.log %11 : (tensor<99x46x97xf32>) -> tensor<99x46x97xf32>
    %13 = tosa.reduce_max %12 {axis = 1 : i32} : (tensor<99x46x97xf32>) -> tensor<99x1x97xf32>
    %14 = tosa.reduce_min %5 {axis = 3 : i32} : (tensor<41x6x20x59xi1>) -> tensor<41x6x20x1xi1>
    %15 = tosa.reduce_min %14 {axis = 3 : i32} : (tensor<41x6x20x1xi1>) -> tensor<41x6x20x1xi1>
    %16 = tosa.logical_or %15, %14 : (tensor<41x6x20x1xi1>, tensor<41x6x20x1xi1>) -> tensor<41x6x20x1xi1>
    %17 = tosa.reduce_min %14 {axis = 3 : i32} : (tensor<41x6x20x1xi1>) -> tensor<41x6x20x1xi1>
    %18 = tosa.bitwise_not %17 : (tensor<41x6x20x1xi1>) -> tensor<41x6x20x1xi1>
    %19 = tosa.reduce_max %13 {axis = 1 : i32} : (tensor<99x1x97xf32>) -> tensor<99x1x97xf32>
    %in_zp_20 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_20 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %20 = tosa.negate %18, %in_zp_20, %out_zp_20 : (tensor<41x6x20x1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<41x6x20x1xi1>
    %21 = tosa.log %19 : (tensor<99x1x97xf32>) -> tensor<99x1x97xf32>
    %22 = tosa.clz %18 : (tensor<41x6x20x1xi1>) -> tensor<41x6x20x1xi1>
    %23 = tosa.floor %21 : (tensor<99x1x97xf32>) -> tensor<99x1x97xf32>
    %24 = tosa.logical_right_shift %20, %15 : (tensor<41x6x20x1xi1>, tensor<41x6x20x1xi1>) -> tensor<41x6x20x1xi1>
    %25 = tosa.greater_equal %19, %19 : (tensor<99x1x97xf32>, tensor<99x1x97xf32>) -> tensor<99x1x97xi1>
    return %4, %8, %10, %16, %22, %23, %24, %25 : tensor<95x9x45xi16>, tensor<41x6x20x59xi1>, tensor<99x46x97xi1>, tensor<41x6x20x1xi1>, tensor<41x6x20x1xi1>, tensor<99x1x97xf32>, tensor<41x6x20x1xi1>, tensor<99x1x97xi1>
  }
}
