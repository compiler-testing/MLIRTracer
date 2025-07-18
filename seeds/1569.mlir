module {
  func.func @main(%arg0: tensor<39x87x18xf32>, %arg1: tensor<1x1x18xf32>, %arg2: tensor<9x43x44x3x67x69xi32>, %arg3: tensor<9x43x1x1x67x1xi32>, %arg4: tensor<41x8x54x44xf32>, %arg5: tensor<54x86x98x81xf32>, %arg6: tensor<54xf32>, %arg7: tensor<5x88xi1>, %arg8: tensor<5x1xi1>) -> (tensor<9x43x44x3x67x69xi32>, tensor<41x104x207x1xf32>, tensor<5x88xi1>, tensor<39x87x18xi1>, tensor<41x104x207x54xf32>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<39x87x18xf32>, tensor<1x1x18xf32>) -> tensor<39x87x18xf32>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<9x43x44x3x67x69xi32>, tensor<9x43x1x1x67x1xi32>) -> tensor<9x43x44x3x67x69xi32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg4, %arg5, %arg6, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 2, 2, 1>, stride = array<i64: 2, 2>, out_shape = array<i64: 41, 104, 207, 54>} : (tensor<41x8x54x44xf32>, tensor<54x86x98x81xf32>, tensor<54xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<41x104x207x54xf32>
    %3 = tosa.reduce_sum %2 {axis = 3 : i32} : (tensor<41x104x207x54xf32>) -> tensor<41x104x207x1xf32>
    %4 = tosa.logical_xor %arg7, %arg8 : (tensor<5x88xi1>, tensor<5x1xi1>) -> tensor<5x88xi1>
    %5 = tosa.greater %0, %0 : (tensor<39x87x18xf32>, tensor<39x87x18xf32>) -> tensor<39x87x18xi1>
    %6 = tosa.floor %2 : (tensor<41x104x207x54xf32>) -> tensor<41x104x207x54xf32>
    return %1, %3, %4, %5, %6 : tensor<9x43x44x3x67x69xi32>, tensor<41x104x207x1xf32>, tensor<5x88xi1>, tensor<39x87x18xi1>, tensor<41x104x207x54xf32>
  }
}
