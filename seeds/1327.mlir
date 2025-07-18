module {
  func.func @main(%arg0: tensor<27x80x32x80xi32>, %arg1: tensor<1x80x32x1xi32>, %arg2: tensor<f32>, %arg3: tensor<92x58x82x58xf32>, %arg4: tensor<1x58x82x58xf32>, %arg5: tensor<25x95x83x13x98x40xi32>, %arg6: tensor<25x1x83x13x98x1xi32>) -> (tensor<8x9x11x5xi1>, tensor<1x80x1x80xi1>, tensor<27x1x1x80xi1>, tensor<f32>, tensor<92x58x82x58xf32>, tensor<i1>, tensor<92x58x82x58xf32>, tensor<25x95x83x13x98x40xi32>, tensor<27x1x1x80xi1>, tensor<f32>, tensor<25x95x83x13x98x40xi32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<27x80x32x80xi32>, tensor<1x80x32x1xi32>) -> tensor<27x80x32x80xi1>
    %1 = tosa.log %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.reduce_min %0 {axis = 2 : i32} : (tensor<27x80x32x80xi1>) -> tensor<27x80x1x80xi1>
    %3 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<27x80x1x80xi1>) -> tensor<1x80x1x80xi1>
    %4 = tosa.reduce_any %3 {axis = 3 : i32} : (tensor<1x80x1x80xi1>) -> tensor<1x80x1x1xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 0, 1, 0, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_5_size = tosa.const_shape {values = dense<[ 8, 9, 11, 5 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<1x80x1x1xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<8x9x11x5xi1>
    %6 = tosa.log %1 : (tensor<f32>) -> tensor<f32>
    %7 = tosa.reduce_product %2 {axis = 1 : i32} : (tensor<27x80x1x80xi1>) -> tensor<27x1x1x80xi1>
    %8 = tosa.arithmetic_right_shift %2, %2 {round = true} : (tensor<27x80x1x80xi1>, tensor<27x80x1x80xi1>) -> tensor<27x80x1x80xi1>
    %9 = tosa.logical_or %8, %2 : (tensor<27x80x1x80xi1>, tensor<27x80x1x80xi1>) -> tensor<27x80x1x80xi1>
    %10 = tosa.reduce_max %9 {axis = 0 : i32} : (tensor<27x80x1x80xi1>) -> tensor<1x80x1x80xi1>
    %11 = tosa.floor %6 : (tensor<f32>) -> tensor<f32>
    %12 = tosa.logical_xor %9, %8 : (tensor<27x80x1x80xi1>, tensor<27x80x1x80xi1>) -> tensor<27x80x1x80xi1>
    %13 = tosa.logical_xor %10, %10 : (tensor<1x80x1x80xi1>, tensor<1x80x1x80xi1>) -> tensor<1x80x1x80xi1>
    %14 = tosa.reverse %7 {axis = 1 : i32} : (tensor<27x1x1x80xi1>) -> tensor<27x1x1x80xi1>
    %15 = tosa.equal %1, %6 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %in_zp_16 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_16 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %16 = tosa.negate %11, %in_zp_16, %out_zp_16 : (tensor<f32>, tensor<1xf32>, tensor<1xf32>) -> tensor<f32>
    %17 = tosa.minimum %arg3, %arg4 : (tensor<92x58x82x58xf32>, tensor<1x58x82x58xf32>) -> tensor<92x58x82x58xf32>
    %18 = tosa.reverse %17 {axis = 0 : i32} : (tensor<92x58x82x58xf32>) -> tensor<92x58x82x58xf32>
    %19 = tosa.intdiv %arg5, %arg6 : (tensor<25x95x83x13x98x40xi32>, tensor<25x1x83x13x98x1xi32>) -> tensor<25x95x83x13x98x40xi32>
    %20 = tosa.arithmetic_right_shift %15, %15 {round = true} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %21 = tosa.exp %17 : (tensor<92x58x82x58xf32>) -> tensor<92x58x82x58xf32>
    %22 = tosa.intdiv %19, %19 : (tensor<25x95x83x13x98x40xi32>, tensor<25x95x83x13x98x40xi32>) -> tensor<25x95x83x13x98x40xi32>
    %23 = tosa.bitwise_xor %19, %19 : (tensor<25x95x83x13x98x40xi32>, tensor<25x95x83x13x98x40xi32>) -> tensor<25x95x83x13x98x40xi32>
    %24 = tosa.minimum %23, %23 : (tensor<25x95x83x13x98x40xi32>, tensor<25x95x83x13x98x40xi32>) -> tensor<25x95x83x13x98x40xi32>
    %25 = tosa.reduce_any %12 {axis = 1 : i32} : (tensor<27x80x1x80xi1>) -> tensor<27x1x1x80xi1>
    %26 = tosa.exp %6 : (tensor<f32>) -> tensor<f32>
    %27 = tosa.logical_left_shift %23, %22 : (tensor<25x95x83x13x98x40xi32>, tensor<25x95x83x13x98x40xi32>) -> tensor<25x95x83x13x98x40xi32>
    return %5, %13, %14, %16, %18, %20, %21, %24, %25, %26, %27 : tensor<8x9x11x5xi1>, tensor<1x80x1x80xi1>, tensor<27x1x1x80xi1>, tensor<f32>, tensor<92x58x82x58xf32>, tensor<i1>, tensor<92x58x82x58xf32>, tensor<25x95x83x13x98x40xi32>, tensor<27x1x1x80xi1>, tensor<f32>, tensor<25x95x83x13x98x40xi32>
  }
}
