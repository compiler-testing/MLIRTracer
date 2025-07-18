module {
  func.func @main(%arg0: tensor<14x57x70xi32>, %arg1: tensor<1x57x70xi32>, %arg2: tensor<99x21x36xf32>, %arg3: tensor<i1>, %arg4: tensor<i1>, %arg5: tensor<5x87x37x16xi1>) -> (tensor<14x57x70xi32>, tensor<1x1x36xf32>, tensor<i1>, tensor<11x2x6xf32>, tensor<11x2x3xf32>, tensor<99x21x36xf32>, tensor<5x1x1xi32>, tensor<5x87x1x16xi1>, tensor<16x5x87x1xi1>, tensor<10x1x1xi32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<14x57x70xi32>, tensor<1x57x70xi32>) -> tensor<14x57x70xi32>
    %1 = tosa.intdiv %0, %0 : (tensor<14x57x70xi32>, tensor<14x57x70xi32>) -> tensor<14x57x70xi32>
    %2 = tosa.log %arg2 : (tensor<99x21x36xf32>) -> tensor<99x21x36xf32>
    %3 = tosa.log %2 : (tensor<99x21x36xf32>) -> tensor<99x21x36xf32>
    %4 = tosa.clamp %3 {min_val = -4.200000e+01 : f32, max_val = 1.350000e+02 : f32} : (tensor<99x21x36xf32>) -> tensor<99x21x36xf32>
    %5 = tosa.reduce_max %4 {axis = 1 : i32} : (tensor<99x21x36xf32>) -> tensor<99x1x36xf32>
    %6 = tosa.logical_xor %arg3, %arg4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %s_7_start = tosa.const_shape {values = dense<[ 88, 19, 33 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_7_size = tosa.const_shape {values = dense<[ 11, 2, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = tosa.slice %2, %s_7_start, %s_7_size : (tensor<99x21x36xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<11x2x3xf32>
    %8 = tosa.reduce_min %5 {axis = 0 : i32} : (tensor<99x1x36xf32>) -> tensor<1x1x36xf32>
    %9 = tosa.logical_or %6, %6 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %10 = tosa.concat %7, %7 {axis = 2 : i32} : (tensor<11x2x3xf32>, tensor<11x2x3xf32>) -> tensor<11x2x6xf32>
    %11 = tosa.reduce_any %arg5 {axis = 2 : i32} : (tensor<5x87x37x16xi1>) -> tensor<5x87x1x16xi1>
    %12 = tosa.argmax %11 {axis = 1 : i32} : (tensor<5x87x1x16xi1>) -> tensor<5x1x16xi32>
    %13 = tosa.sigmoid %7 : (tensor<11x2x3xf32>) -> tensor<11x2x3xf32>
    %14 = tosa.maximum %12, %12 : (tensor<5x1x16xi32>, tensor<5x1x16xi32>) -> tensor<5x1x16xi32>
    %15 = tosa.reduce_all %11 {axis = 2 : i32} : (tensor<5x87x1x16xi1>) -> tensor<5x87x1x16xi1>
    %16 = tosa.abs %14 : (tensor<5x1x16xi32>) -> tensor<5x1x16xi32>
    %17 = tosa.reduce_min %16 {axis = 2 : i32} : (tensor<5x1x16xi32>) -> tensor<5x1x1xi32>
    %18 = tosa.logical_left_shift %17, %17 : (tensor<5x1x1xi32>, tensor<5x1x1xi32>) -> tensor<5x1x1xi32>
    %19 = tosa.log %3 : (tensor<99x21x36xf32>) -> tensor<99x21x36xf32>
    %20 = tosa.logical_right_shift %17, %17 : (tensor<5x1x1xi32>, tensor<5x1x1xi32>) -> tensor<5x1x1xi32>
    %21 = tosa.logical_or %15, %11 : (tensor<5x87x1x16xi1>, tensor<5x87x1x16xi1>) -> tensor<5x87x1x16xi1>
    %22 = "tosa.const"() {values = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %23 = tosa.transpose %11 {perms = array<i32: 3, 0, 1, 2>} : (tensor<5x87x1x16xi1>) -> tensor<16x5x87x1xi1>
    %24 = tosa.intdiv %17, %18 : (tensor<5x1x1xi32>, tensor<5x1x1xi32>) -> tensor<5x1x1xi32>
    %25 = tosa.concat %24, %18 {axis = 0 : i32} : (tensor<5x1x1xi32>, tensor<5x1x1xi32>) -> tensor<10x1x1xi32>
    return %1, %8, %9, %10, %13, %19, %20, %21, %23, %25 : tensor<14x57x70xi32>, tensor<1x1x36xf32>, tensor<i1>, tensor<11x2x6xf32>, tensor<11x2x3xf32>, tensor<99x21x36xf32>, tensor<5x1x1xi32>, tensor<5x87x1x16xi1>, tensor<16x5x87x1xi1>, tensor<10x1x1xi32>
  }
}
