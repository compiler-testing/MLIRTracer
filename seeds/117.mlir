module {
  func.func @main(%arg0: tensor<46x16x59x5x68xi16>, %arg1: tensor<99x37x11x4x77xi1>, %arg2: tensor<63x83x18xi64>, %arg3: tensor<1x1x18xi64>, %arg4: tensor<55x78x75xf32>, %arg5: tensor<1x1x1xf32>) -> (tensor<68x5x16x59x46xi16>, tensor<63x83x18xi1>, tensor<99x37x11x4x77xi1>, tensor<99x37x11x4x77xi1>, tensor<55x78x75xf32>, tensor<1x78x75xf32>, tensor<55x78x75xf32>, tensor<63x18xi32>) {
    %0 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<46x16x59x5x68xi16>) -> tensor<68x5x16x59x46xi16>
    %2 = tosa.logical_not %arg1 : (tensor<99x37x11x4x77xi1>) -> tensor<99x37x11x4x77xi1>
    %3 = tosa.minimum %arg2, %arg3 : (tensor<63x83x18xi64>, tensor<1x1x18xi64>) -> tensor<63x83x18xi64>
    %4 = tosa.logical_xor %2, %2 : (tensor<99x37x11x4x77xi1>, tensor<99x37x11x4x77xi1>) -> tensor<99x37x11x4x77xi1>
    %5 = tosa.arithmetic_right_shift %2, %2 {round = false} : (tensor<99x37x11x4x77xi1>, tensor<99x37x11x4x77xi1>) -> tensor<99x37x11x4x77xi1>
    %6 = tosa.sub %2, %4 : (tensor<99x37x11x4x77xi1>, tensor<99x37x11x4x77xi1>) -> tensor<99x37x11x4x77xi1>
    %7 = tosa.greater_equal %3, %3 : (tensor<63x83x18xi64>, tensor<63x83x18xi64>) -> tensor<63x83x18xi1>
    %8 = tosa.argmax %3 {axis = 1 : i32} : (tensor<63x83x18xi64>) -> tensor<63x18xi32>
    %9 = tosa.bitwise_xor %6, %5 : (tensor<99x37x11x4x77xi1>, tensor<99x37x11x4x77xi1>) -> tensor<99x37x11x4x77xi1>
    %10 = tosa.minimum %8, %8 : (tensor<63x18xi32>, tensor<63x18xi32>) -> tensor<63x18xi32>
    %11 = tosa.pow %arg4, %arg5 : (tensor<55x78x75xf32>, tensor<1x1x1xf32>) -> tensor<55x78x75xf32>
    %12 = tosa.bitwise_and %4, %4 : (tensor<99x37x11x4x77xi1>, tensor<99x37x11x4x77xi1>) -> tensor<99x37x11x4x77xi1>
    %in_zp_13 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_13 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %13 = tosa.negate %12, %in_zp_13, %out_zp_13 : (tensor<99x37x11x4x77xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<99x37x11x4x77xi1>
    %14 = tosa.bitwise_or %9, %4 : (tensor<99x37x11x4x77xi1>, tensor<99x37x11x4x77xi1>) -> tensor<99x37x11x4x77xi1>
    %15 = tosa.bitwise_xor %9, %14 : (tensor<99x37x11x4x77xi1>, tensor<99x37x11x4x77xi1>) -> tensor<99x37x11x4x77xi1>
    %16 = tosa.pow %11, %11 : (tensor<55x78x75xf32>, tensor<55x78x75xf32>) -> tensor<55x78x75xf32>
    %17 = tosa.ceil %11 : (tensor<55x78x75xf32>) -> tensor<55x78x75xf32>
    %18 = tosa.clamp %11 {min_val = -4.700000e+01 : f32, max_val = -1.300000e+01 : f32} : (tensor<55x78x75xf32>) -> tensor<55x78x75xf32>
    %19 = tosa.sub %18, %11 : (tensor<55x78x75xf32>, tensor<55x78x75xf32>) -> tensor<55x78x75xf32>
    %in_zp_20 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_20 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %20 = tosa.negate %17, %in_zp_20, %out_zp_20 : (tensor<55x78x75xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<55x78x75xf32>
    %21 = tosa.reduce_product %20 {axis = 0 : i32} : (tensor<55x78x75xf32>) -> tensor<1x78x75xf32>
    %in_zp_22 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_22 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %22 = tosa.negate %16, %in_zp_22, %out_zp_22 : (tensor<55x78x75xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<55x78x75xf32>
    %23 = tosa.bitwise_and %10, %10 : (tensor<63x18xi32>, tensor<63x18xi32>) -> tensor<63x18xi32>
    return %1, %7, %13, %15, %19, %21, %22, %23 : tensor<68x5x16x59x46xi16>, tensor<63x83x18xi1>, tensor<99x37x11x4x77xi1>, tensor<99x37x11x4x77xi1>, tensor<55x78x75xf32>, tensor<1x78x75xf32>, tensor<55x78x75xf32>, tensor<63x18xi32>
  }
}
