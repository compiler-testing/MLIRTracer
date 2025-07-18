module {
  func.func @main(%arg0: tensor<49x36xi1>, %arg1: tensor<49x1xi1>, %arg2: tensor<10x77x34x33xf32>, %arg3: tensor<92x52x10x44xf32>, %arg4: tensor<92xf32>) -> (tensor<10x207x46x92xf32>, tensor<49x36xi1>, tensor<49x36xi1>, tensor<9x8x7x4xf32>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<49x36xi1>, tensor<49x1xi1>) -> tensor<49x36xi1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<49x36xi1>, tensor<49x36xi1>) -> tensor<49x36xi1>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 2, 1, 2, 1>, stride = array<i64: 2, 1>, out_shape = array<i64: 10, 207, 46, 92>} : (tensor<10x77x34x33xf32>, tensor<92x52x10x44xf32>, tensor<92xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10x207x46x92xf32>
    %3 = tosa.logical_and %1, %0 : (tensor<49x36xi1>, tensor<49x36xi1>) -> tensor<49x36xi1>
    %4 = tosa.tanh %2 : (tensor<10x207x46x92xf32>) -> tensor<10x207x46x92xf32>
    %5 = tosa.pow %4, %2 : (tensor<10x207x46x92xf32>, tensor<10x207x46x92xf32>) -> tensor<10x207x46x92xf32>
    %6 = tosa.maximum %5, %2 : (tensor<10x207x46x92xf32>, tensor<10x207x46x92xf32>) -> tensor<10x207x46x92xf32>
    %7 = tosa.logical_left_shift %3, %1 : (tensor<49x36xi1>, tensor<49x36xi1>) -> tensor<49x36xi1>
    %8 = tosa.minimum %2, %4 : (tensor<10x207x46x92xf32>, tensor<10x207x46x92xf32>) -> tensor<10x207x46x92xf32>
    %9 = tosa.clz %3 : (tensor<49x36xi1>) -> tensor<49x36xi1>
    %s_10_start = tosa.const_shape {values = dense<[ 1, 9, 3, 4 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_10_size = tosa.const_shape {values = dense<[ 9, 8, 7, 4 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.slice %8, %s_10_start, %s_10_size : (tensor<10x207x46x92xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<9x8x7x4xf32>
    return %6, %7, %9, %10 : tensor<10x207x46x92xf32>, tensor<49x36xi1>, tensor<49x36xi1>, tensor<9x8x7x4xf32>
  }
}
