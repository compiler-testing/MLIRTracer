module {
  func.func @main(%arg0: tensor<11xi32>, %arg1: tensor<84xi32>, %arg2: tensor<8x97x80x12x30xf32>) -> (tensor<2x1x5587200x1xi1>, tensor<2x1x5587200x2xi1>, tensor<95xi32>, tensor<1x1x5587200x2xf32>, tensor<2x2x5587200x2xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<11xi32>, tensor<84xi32>) -> tensor<95xi32>
    %1 = tosa.sigmoid %arg2 : (tensor<8x97x80x12x30xf32>) -> tensor<8x97x80x12x30xf32>
    %r_2 = tosa.const_shape {values = dense<[ 2, 1, 5587200, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %1, %r_2 : (tensor<8x97x80x12x30xf32>, !tosa.shape<4>) -> tensor<2x1x5587200x2xf32>
    %3 = tosa.floor %2 : (tensor<2x1x5587200x2xf32>) -> tensor<2x1x5587200x2xf32>
    %4 = tosa.exp %2 : (tensor<2x1x5587200x2xf32>) -> tensor<2x1x5587200x2xf32>
    %5 = tosa.clamp %4 {min_val = -4.400000e+01 : f32, max_val = 9.000000e+00 : f32} : (tensor<2x1x5587200x2xf32>) -> tensor<2x1x5587200x2xf32>
    %6 = tosa.greater %3, %4 : (tensor<2x1x5587200x2xf32>, tensor<2x1x5587200x2xf32>) -> tensor<2x1x5587200x2xi1>
    %7 = tosa.arithmetic_right_shift %6, %6 {round = false} : (tensor<2x1x5587200x2xi1>, tensor<2x1x5587200x2xi1>) -> tensor<2x1x5587200x2xi1>
    %8 = tosa.reduce_product %7 {axis = 3 : i32} : (tensor<2x1x5587200x2xi1>) -> tensor<2x1x5587200x1xi1>
    %9 = tosa.log %5 : (tensor<2x1x5587200x2xf32>) -> tensor<2x1x5587200x2xf32>
    %10 = tosa.reverse %9 {axis = 1 : i32} : (tensor<2x1x5587200x2xf32>) -> tensor<2x1x5587200x2xf32>
    %11 = tosa.reduce_any %6 {axis = 1 : i32} : (tensor<2x1x5587200x2xi1>) -> tensor<2x1x5587200x2xi1>
    %12 = tosa.bitwise_xor %0, %0 : (tensor<95xi32>, tensor<95xi32>) -> tensor<95xi32>
    %13 = tosa.pow %5, %10 : (tensor<2x1x5587200x2xf32>, tensor<2x1x5587200x2xf32>) -> tensor<2x1x5587200x2xf32>
    %14 = tosa.reduce_min %10 {axis = 0 : i32} : (tensor<2x1x5587200x2xf32>) -> tensor<1x1x5587200x2xf32>
    %15 = tosa.greater_equal %13, %13 : (tensor<2x1x5587200x2xf32>, tensor<2x1x5587200x2xf32>) -> tensor<2x1x5587200x2xi1>
    %16 = tosa.maximum %14, %14 : (tensor<1x1x5587200x2xf32>, tensor<1x1x5587200x2xf32>) -> tensor<1x1x5587200x2xf32>
    %17 = tosa.reduce_min %16 {axis = 1 : i32} : (tensor<1x1x5587200x2xf32>) -> tensor<1x1x5587200x2xf32>
    %18 = tosa.concat %15, %7 {axis = 1 : i32} : (tensor<2x1x5587200x2xi1>, tensor<2x1x5587200x2xi1>) -> tensor<2x2x5587200x2xi1>
    return %8, %11, %12, %17, %18 : tensor<2x1x5587200x1xi1>, tensor<2x1x5587200x2xi1>, tensor<95xi32>, tensor<1x1x5587200x2xf32>, tensor<2x2x5587200x2xi1>
  }
}
