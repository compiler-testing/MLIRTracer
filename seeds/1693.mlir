module {
  func.func @main(%arg0: tensor<77x4xi16>, %arg1: tensor<52x65x84x18x90xf32>, %arg2: tensor<52x1x84x18x1xf32>, %arg3: tensor<56x65xf32>) -> (tensor<52x65x84x18x90xi1>, tensor<1x65xf32>, tensor<65xi32>, tensor<231x8xi16>, tensor<56x65xf32>, tensor<56x65xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.tile %arg0, %t_0 : (tensor<77x4xi16>, !tosa.shape<2>) -> tensor<231x8xi16>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<231x8xi16>, tensor<231x8xi16>) -> tensor<231x8xi16>
    %2 = tosa.equal %arg1, %arg2 : (tensor<52x65x84x18x90xf32>, tensor<52x1x84x18x1xf32>) -> tensor<52x65x84x18x90xi1>
    %3 = tosa.logical_left_shift %1, %1 : (tensor<231x8xi16>, tensor<231x8xi16>) -> tensor<231x8xi16>
    %4 = tosa.sigmoid %arg3 : (tensor<56x65xf32>) -> tensor<56x65xf32>
    %5 = tosa.reduce_product %4 {axis = 0 : i32} : (tensor<56x65xf32>) -> tensor<1x65xf32>
    %6 = tosa.exp %5 : (tensor<1x65xf32>) -> tensor<1x65xf32>
    %7 = tosa.log %6 : (tensor<1x65xf32>) -> tensor<1x65xf32>
    %8 = tosa.argmax %4 {axis = 0 : i32} : (tensor<56x65xf32>) -> tensor<65xi32>
    %9 = tosa.abs %3 : (tensor<231x8xi16>) -> tensor<231x8xi16>
    %10 = tosa.logical_left_shift %9, %1 : (tensor<231x8xi16>, tensor<231x8xi16>) -> tensor<231x8xi16>
    %11 = tosa.add %4, %4 : (tensor<56x65xf32>, tensor<56x65xf32>) -> tensor<56x65xf32>
    %in_zp_12 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_12 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %12 = tosa.negate %11, %in_zp_12, %out_zp_12 : (tensor<56x65xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<56x65xf32>
    %13 = tosa.add %11, %11 : (tensor<56x65xf32>, tensor<56x65xf32>) -> tensor<56x65xf32>
    return %2, %7, %8, %10, %12, %13 : tensor<52x65x84x18x90xi1>, tensor<1x65xf32>, tensor<65xi32>, tensor<231x8xi16>, tensor<56x65xf32>, tensor<56x65xf32>
  }
}
