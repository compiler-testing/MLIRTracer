module {
  func.func @main(%arg0: tensor<56x35x59x71x95x52xi1>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<52x48x44x80x65xf32>, %arg4: tensor<18x36x93x74xi1>) -> (tensor<56x35x59x71x95x52xi1>, tensor<i1>, tensor<18x1x93x74xi1>, tensor<52x48x44x80x65xi1>, tensor<52x48x44x80x65xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<56x35x59x71x95x52xi1>) -> tensor<56x35x59x71x95x52xi1>
    %1 = tosa.logical_not %0 : (tensor<56x35x59x71x95x52xi1>) -> tensor<56x35x59x71x95x52xi1>
    %2 = tosa.logical_xor %1, %1 : (tensor<56x35x59x71x95x52xi1>, tensor<56x35x59x71x95x52xi1>) -> tensor<56x35x59x71x95x52xi1>
    %3 = tosa.greater_equal %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = tosa.exp %arg3 : (tensor<52x48x44x80x65xf32>) -> tensor<52x48x44x80x65xf32>
    %5 = tosa.reduce_all %arg4 {axis = 1 : i32} : (tensor<18x36x93x74xi1>) -> tensor<18x1x93x74xi1>
    %6 = tosa.greater %4, %4 : (tensor<52x48x44x80x65xf32>, tensor<52x48x44x80x65xf32>) -> tensor<52x48x44x80x65xi1>
    %7 = tosa.arithmetic_right_shift %5, %5 {round = true} : (tensor<18x1x93x74xi1>, tensor<18x1x93x74xi1>) -> tensor<18x1x93x74xi1>
    %8 = tosa.bitwise_or %6, %6 : (tensor<52x48x44x80x65xi1>, tensor<52x48x44x80x65xi1>) -> tensor<52x48x44x80x65xi1>
    %in_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_9 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %9 = tosa.negate %8, %in_zp_9, %out_zp_9 : (tensor<52x48x44x80x65xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<52x48x44x80x65xi1>
    %10 = tosa.pow %4, %4 : (tensor<52x48x44x80x65xf32>, tensor<52x48x44x80x65xf32>) -> tensor<52x48x44x80x65xf32>
    return %2, %3, %7, %9, %10 : tensor<56x35x59x71x95x52xi1>, tensor<i1>, tensor<18x1x93x74xi1>, tensor<52x48x44x80x65xi1>, tensor<52x48x44x80x65xf32>
  }
}
