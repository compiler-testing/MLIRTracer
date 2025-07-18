module {
  func.func @main(%arg0: tensor<27x82x97xi16>, %arg1: tensor<83xf32>, %arg2: tensor<1xf32>, %arg3: tensor<3x100x22x92x21xi32>, %arg4: tensor<3x100x1x92x21xi32>, %arg5: tensor<80x29x56x42x44xi1>, %arg6: tensor<1x1x1x1x44xi1>) -> (tensor<27x82x97xi16>, tensor<3x100x22x92x21xi32>, tensor<1xf32>, tensor<4466x16x3360xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<27x82x97xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<27x82x97xi16>
    %1 = tosa.minimum %arg1, %arg2 : (tensor<83xf32>, tensor<1xf32>) -> tensor<83xf32>
    %2 = tosa.intdiv %arg3, %arg4 : (tensor<3x100x22x92x21xi32>, tensor<3x100x1x92x21xi32>) -> tensor<3x100x22x92x21xi32>
    %3 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<27x82x97xi16>, tensor<27x82x97xi16>) -> tensor<27x82x97xi16>
    %4 = tosa.pow %1, %1 : (tensor<83xf32>, tensor<83xf32>) -> tensor<83xf32>
    %5 = tosa.pow %4, %1 : (tensor<83xf32>, tensor<83xf32>) -> tensor<83xf32>
    %6 = tosa.reduce_max %5 {axis = 0 : i32} : (tensor<83xf32>) -> tensor<1xf32>
    %7 = tosa.logical_xor %arg5, %arg6 : (tensor<80x29x56x42x44xi1>, tensor<1x1x1x1x44xi1>) -> tensor<80x29x56x42x44xi1>
    %8 = tosa.exp %6 : (tensor<1xf32>) -> tensor<1xf32>
    %9 = tosa.maximum %2, %2 : (tensor<3x100x22x92x21xi32>, tensor<3x100x22x92x21xi32>) -> tensor<3x100x22x92x21xi32>
    %10 = tosa.log %8 : (tensor<1xf32>) -> tensor<1xf32>
    %r_11 = tosa.const_shape {values = dense<[ 4466, 16, 3360 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %11 = tosa.reshape %7, %r_11 : (tensor<80x29x56x42x44xi1>, !tosa.shape<3>) -> tensor<4466x16x3360xi1>
    return %3, %9, %10, %11 : tensor<27x82x97xi16>, tensor<3x100x22x92x21xi32>, tensor<1xf32>, tensor<4466x16x3360xi1>
  }
}
