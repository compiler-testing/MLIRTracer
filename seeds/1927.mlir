module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<65x90x3x34x51x33xi32>, %arg2: tensor<1x1x3x1x51x1xi32>, %arg3: tensor<70x7x12x24xi1>) -> (tensor<1x1x1x1xi16>, tensor<65x90x6x34x51x33xi1>, tensor<1x7x12x24xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<i16>, tensor<1xi16>, tensor<1xi16>) -> tensor<i16>
    %r_1 = tosa.const_shape {values = dense<[ 1, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.reshape %0, %r_1 : (tensor<i16>, !tosa.shape<4>) -> tensor<1x1x1x1xi16>
    %2 = tosa.reverse %1 {axis = 2 : i32} : (tensor<1x1x1x1xi16>) -> tensor<1x1x1x1xi16>
    %3 = tosa.intdiv %arg1, %arg2 : (tensor<65x90x3x34x51x33xi32>, tensor<1x1x3x1x51x1xi32>) -> tensor<65x90x3x34x51x33xi32>
    %4 = tosa.greater %3, %3 : (tensor<65x90x3x34x51x33xi32>, tensor<65x90x3x34x51x33xi32>) -> tensor<65x90x3x34x51x33xi1>
    %5 = tosa.reduce_all %arg3 {axis = 0 : i32} : (tensor<70x7x12x24xi1>) -> tensor<1x7x12x24xi1>
    %6 = tosa.concat %4, %4 {axis = 2 : i32} : (tensor<65x90x3x34x51x33xi1>, tensor<65x90x3x34x51x33xi1>) -> tensor<65x90x6x34x51x33xi1>
    %7 = tosa.logical_left_shift %6, %6 : (tensor<65x90x6x34x51x33xi1>, tensor<65x90x6x34x51x33xi1>) -> tensor<65x90x6x34x51x33xi1>
    %8 = tosa.logical_right_shift %5, %5 : (tensor<1x7x12x24xi1>, tensor<1x7x12x24xi1>) -> tensor<1x7x12x24xi1>
    %9 = tosa.abs %8 : (tensor<1x7x12x24xi1>) -> tensor<1x7x12x24xi1>
    return %2, %7, %9 : tensor<1x1x1x1xi16>, tensor<65x90x6x34x51x33xi1>, tensor<1x7x12x24xi1>
  }
}
