module {
  func.func @main(%arg0: tensor<99x64x44x87xf32>, %arg1: tensor<79x47x46x78xf32>, %arg2: tensor<79xf32>) -> (tensor<99x176x135x79xi1>, tensor<99x176x1x79xi1>, tensor<9x11x4xi1>, tensor<99x176x135x79xf32>, tensor<99x176x135x79xf32>, tensor<99x176x135x79xf32>, tensor<99x135x79xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 2, 1, 1, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 99, 176, 135, 79>} : (tensor<99x64x44x87xf32>, tensor<79x47x46x78xf32>, tensor<79xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<99x176x135x79xf32>
    %1 = tosa.greater_equal %0, %0 : (tensor<99x176x135x79xf32>, tensor<99x176x135x79xf32>) -> tensor<99x176x135x79xi1>
    %2 = tosa.sigmoid %0 : (tensor<99x176x135x79xf32>) -> tensor<99x176x135x79xf32>
    %3 = tosa.argmax %1 {axis = 1 : i32} : (tensor<99x176x135x79xi1>) -> tensor<99x135x79xi32>
    %4 = tosa.abs %3 : (tensor<99x135x79xi32>) -> tensor<99x135x79xi32>
    %5 = tosa.logical_or %1, %1 : (tensor<99x176x135x79xi1>, tensor<99x176x135x79xi1>) -> tensor<99x176x135x79xi1>
    %6 = tosa.reduce_all %1 {axis = 2 : i32} : (tensor<99x176x135x79xi1>) -> tensor<99x176x1x79xi1>
    %7 = tosa.abs %4 : (tensor<99x135x79xi32>) -> tensor<99x135x79xi32>
    %s_8_start = tosa.const_shape {values = dense<[ 40, 10, 52 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_8_size = tosa.const_shape {values = dense<[ 9, 11, 4 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %8 = tosa.slice %7, %s_8_start, %s_8_size : (tensor<99x135x79xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<9x11x4xi32>
    %9 = tosa.greater_equal %8, %8 : (tensor<9x11x4xi32>, tensor<9x11x4xi32>) -> tensor<9x11x4xi1>
    %10 = tosa.log %2 : (tensor<99x176x135x79xf32>) -> tensor<99x176x135x79xf32>
    %11 = tosa.log %2 : (tensor<99x176x135x79xf32>) -> tensor<99x176x135x79xf32>
    %12 = tosa.exp %2 : (tensor<99x176x135x79xf32>) -> tensor<99x176x135x79xf32>
    %13 = tosa.greater_equal %4, %3 : (tensor<99x135x79xi32>, tensor<99x135x79xi32>) -> tensor<99x135x79xi1>
    return %5, %6, %9, %10, %11, %12, %13 : tensor<99x176x135x79xi1>, tensor<99x176x1x79xi1>, tensor<9x11x4xi1>, tensor<99x176x135x79xf32>, tensor<99x176x135x79xf32>, tensor<99x176x135x79xf32>, tensor<99x135x79xi1>
  }
}
