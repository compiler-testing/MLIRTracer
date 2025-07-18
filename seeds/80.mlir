module {
  func.func @main(%arg0: tensor<99x24x35x79xf32>, %arg1: tensor<15x18x64x49xf32>, %arg2: tensor<15xf32>, %arg3: tensor<25x16xi8>, %arg4: tensor<25x1xi8>) -> (tensor<99x43x134x15xf32>, tensor<25x16xi1>, tensor<25x16xi8>, tensor<25x1xi1>, tensor<25x1xi1>, tensor<1x16xi1>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 2>, out_shape = array<i64: 99, 43, 134, 15>} : (tensor<99x24x35x79xf32>, tensor<15x18x64x49xf32>, tensor<15xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<99x43x134x15xf32>
    %1 = tosa.bitwise_xor %arg3, %arg4 : (tensor<25x16xi8>, tensor<25x1xi8>) -> tensor<25x16xi8>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<25x16xi8>, tensor<25x16xi8>) -> tensor<25x16xi8>
    %3 = tosa.greater %2, %1 : (tensor<25x16xi8>, tensor<25x16xi8>) -> tensor<25x16xi1>
    %4 = tosa.reduce_min %3 {axis = 1 : i32} : (tensor<25x16xi1>) -> tensor<25x1xi1>
    %5 = tosa.greater %1, %1 : (tensor<25x16xi8>, tensor<25x16xi8>) -> tensor<25x16xi1>
    %6 = tosa.maximum %1, %2 : (tensor<25x16xi8>, tensor<25x16xi8>) -> tensor<25x16xi8>
    %7 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<25x16xi8>) -> tensor<1x16xi8>
    %8 = tosa.reduce_all %4 {axis = 1 : i32} : (tensor<25x1xi1>) -> tensor<25x1xi1>
    %9 = tosa.bitwise_or %7, %7 : (tensor<1x16xi8>, tensor<1x16xi8>) -> tensor<1x16xi8>
    %10 = tosa.reduce_all %3 {axis = 0 : i32} : (tensor<25x16xi1>) -> tensor<1x16xi1>
    %11 = tosa.reduce_any %4 {axis = 1 : i32} : (tensor<25x1xi1>) -> tensor<25x1xi1>
    %12 = tosa.greater %9, %9 : (tensor<1x16xi8>, tensor<1x16xi8>) -> tensor<1x16xi1>
    %13 = tosa.logical_not %12 : (tensor<1x16xi1>) -> tensor<1x16xi1>
    %14 = tosa.logical_xor %13, %13 : (tensor<1x16xi1>, tensor<1x16xi1>) -> tensor<1x16xi1>
    %15 = tosa.bitwise_or %14, %10 : (tensor<1x16xi1>, tensor<1x16xi1>) -> tensor<1x16xi1>
    return %0, %5, %6, %8, %11, %15 : tensor<99x43x134x15xf32>, tensor<25x16xi1>, tensor<25x16xi8>, tensor<25x1xi1>, tensor<25x1xi1>, tensor<1x16xi1>
  }
}
