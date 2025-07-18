module {
  func.func @main(%arg0: tensor<99x85x6x10x67xf32>, %arg1: tensor<23x54x23x25xi32>, %arg2: tensor<1x54x23x25xi32>, %arg3: tensor<54x22x66xi1>, %arg4: tensor<1x1x66xi1>) -> (tensor<54x22x66xi1>, tensor<23x1x23x25xi32>, tensor<99x85x6x10x67xf32>, tensor<99x85x6x10x67xf32>, tensor<99x85x6x10x67xf32>, tensor<23x162x46x75xi32>, tensor<99x85x6x10x67xf32>, tensor<99x85x6x10x67xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<99x85x6x10x67xf32>) -> tensor<99x85x6x10x67xf32>
    %1 = tosa.logical_right_shift %arg1, %arg2 : (tensor<23x54x23x25xi32>, tensor<1x54x23x25xi32>) -> tensor<23x54x23x25xi32>
    %2 = tosa.logical_and %arg3, %arg4 : (tensor<54x22x66xi1>, tensor<1x1x66xi1>) -> tensor<54x22x66xi1>
    %3 = tosa.log %0 : (tensor<99x85x6x10x67xf32>) -> tensor<99x85x6x10x67xf32>
    %4 = tosa.sigmoid %3 : (tensor<99x85x6x10x67xf32>) -> tensor<99x85x6x10x67xf32>
    %5 = tosa.bitwise_xor %1, %1 : (tensor<23x54x23x25xi32>, tensor<23x54x23x25xi32>) -> tensor<23x54x23x25xi32>
    %6 = tosa.clamp %5 {min_val = -44 : i32, max_val = -4 : i32} : (tensor<23x54x23x25xi32>) -> tensor<23x54x23x25xi32>
    %7 = tosa.abs %6 : (tensor<23x54x23x25xi32>) -> tensor<23x54x23x25xi32>
    %8 = tosa.intdiv %7, %5 : (tensor<23x54x23x25xi32>, tensor<23x54x23x25xi32>) -> tensor<23x54x23x25xi32>
    %9 = tosa.logical_not %2 : (tensor<54x22x66xi1>) -> tensor<54x22x66xi1>
    %10 = tosa.reduce_product %6 {axis = 1 : i32} : (tensor<23x54x23x25xi32>) -> tensor<23x1x23x25xi32>
    %11 = tosa.sub %8, %5 : (tensor<23x54x23x25xi32>, tensor<23x54x23x25xi32>) -> tensor<23x54x23x25xi32>
    %12 = tosa.log %3 : (tensor<99x85x6x10x67xf32>) -> tensor<99x85x6x10x67xf32>
    %13 = tosa.exp %0 : (tensor<99x85x6x10x67xf32>) -> tensor<99x85x6x10x67xf32>
    %14 = tosa.sigmoid %0 : (tensor<99x85x6x10x67xf32>) -> tensor<99x85x6x10x67xf32>
    %15 = tosa.floor %3 : (tensor<99x85x6x10x67xf32>) -> tensor<99x85x6x10x67xf32>
    %t_16 = tosa.const_shape {values = dense<[ 1, 3, 2, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %16 = tosa.tile %11, %t_16 : (tensor<23x54x23x25xi32>, !tosa.shape<4>) -> tensor<23x162x46x75xi32>
    %17 = tosa.rsqrt %4 : (tensor<99x85x6x10x67xf32>) -> tensor<99x85x6x10x67xf32>
    %18 = tosa.reciprocal %13 : (tensor<99x85x6x10x67xf32>) -> tensor<99x85x6x10x67xf32>
    return %9, %10, %12, %14, %15, %16, %17, %18 : tensor<54x22x66xi1>, tensor<23x1x23x25xi32>, tensor<99x85x6x10x67xf32>, tensor<99x85x6x10x67xf32>, tensor<99x85x6x10x67xf32>, tensor<23x162x46x75xi32>, tensor<99x85x6x10x67xf32>, tensor<99x85x6x10x67xf32>
  }
}
