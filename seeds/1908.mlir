module {
  func.func @main(%arg0: tensor<99x71x23x42xf32>, %arg1: tensor<32x7x1x85xf32>, %arg2: tensor<32xf32>, %arg3: tensor<22x14x64xi32>, %arg4: tensor<1x14x64xi32>, %arg5: tensor<64x27x78x79x5x63xi1>) -> (tensor<64x27x78x79x5x63xi1>, tensor<64x27x78x79x5x63xi1>, tensor<99x79x25x32xi1>, tensor<1x14x64xi1>, tensor<10x2x4xi1>, tensor<99x79x25x32xf32>) {
    %input_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2, %input_zp_0, %weight_zp_0 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 99, 79, 25, 32>} : (tensor<99x71x23x42xf32>, tensor<32x7x1x85xf32>, tensor<32xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<99x79x25x32xf32>
    %1 = tosa.intdiv %arg3, %arg4 : (tensor<22x14x64xi32>, tensor<1x14x64xi32>) -> tensor<22x14x64xi32>
    %2 = tosa.logical_not %arg5 : (tensor<64x27x78x79x5x63xi1>) -> tensor<64x27x78x79x5x63xi1>
    %3 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<22x14x64xi32>) -> tensor<1x14x64xi32>
    %4 = tosa.intdiv %3, %3 : (tensor<1x14x64xi32>, tensor<1x14x64xi32>) -> tensor<1x14x64xi32>
    %5 = tosa.sub %2, %2 : (tensor<64x27x78x79x5x63xi1>, tensor<64x27x78x79x5x63xi1>) -> tensor<64x27x78x79x5x63xi1>
    %6 = tosa.exp %0 : (tensor<99x79x25x32xf32>) -> tensor<99x79x25x32xf32>
    %7 = tosa.greater %4, %3 : (tensor<1x14x64xi32>, tensor<1x14x64xi32>) -> tensor<1x14x64xi1>
    %8 = tosa.add %2, %2 : (tensor<64x27x78x79x5x63xi1>, tensor<64x27x78x79x5x63xi1>) -> tensor<64x27x78x79x5x63xi1>
    %9 = tosa.exp %6 : (tensor<99x79x25x32xf32>) -> tensor<99x79x25x32xf32>
    %10 = tosa.greater %6, %0 : (tensor<99x79x25x32xf32>, tensor<99x79x25x32xf32>) -> tensor<99x79x25x32xi1>
    %11 = tosa.reduce_product %7 {axis = 2 : i32} : (tensor<1x14x64xi1>) -> tensor<1x14x1xi1>
    %12 = tosa.equal %4, %4 : (tensor<1x14x64xi32>, tensor<1x14x64xi32>) -> tensor<1x14x64xi1>
    %s_13_start = tosa.const_shape {values = dense<[ 0, 1, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_13_size = tosa.const_shape {values = dense<[ 10, 2, 4 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %13 = tosa.slice %11, %s_13_start, %s_13_size : (tensor<1x14x1xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<10x2x4xi1>
    %14 = tosa.sigmoid %9 : (tensor<99x79x25x32xf32>) -> tensor<99x79x25x32xf32>
    %15 = tosa.maximum %14, %14 : (tensor<99x79x25x32xf32>, tensor<99x79x25x32xf32>) -> tensor<99x79x25x32xf32>
    return %5, %8, %10, %12, %13, %15 : tensor<64x27x78x79x5x63xi1>, tensor<64x27x78x79x5x63xi1>, tensor<99x79x25x32xi1>, tensor<1x14x64xi1>, tensor<10x2x4xi1>, tensor<99x79x25x32xf32>
  }
}
