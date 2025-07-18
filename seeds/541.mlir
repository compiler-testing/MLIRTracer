module {
  func.func @main(%arg0: tensor<37x46x95x9xi16>, %arg1: tensor<i1>, %arg2: tensor<i1>, %arg3: tensor<22x9x75x6xf32>) -> (tensor<1x9x75x6xf32>, tensor<i1>, tensor<37x92x570x18xi16>, tensor<22x9x75x6xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 1, 1, 3, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<37x46x95x9xi16>, !tosa.shape<4>) -> tensor<37x46x285x9xi16>
    %in_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_1 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %1 = tosa.negate %0, %in_zp_1, %out_zp_1 : (tensor<37x46x285x9xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<37x46x285x9xi16>
    %2 = tosa.arithmetic_right_shift %1, %0 {round = true} : (tensor<37x46x285x9xi16>, tensor<37x46x285x9xi16>) -> tensor<37x46x285x9xi16>
    %t_3 = tosa.const_shape {values = dense<[ 1, 2, 2, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.tile %2, %t_3 : (tensor<37x46x285x9xi16>, !tosa.shape<4>) -> tensor<37x92x570x18xi16>
    %4 = tosa.logical_left_shift %3, %3 : (tensor<37x92x570x18xi16>, tensor<37x92x570x18xi16>) -> tensor<37x92x570x18xi16>
    %5 = tosa.add %4, %3 : (tensor<37x92x570x18xi16>, tensor<37x92x570x18xi16>) -> tensor<37x92x570x18xi16>
    %6 = tosa.logical_or %arg1, %arg2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.log %arg3 : (tensor<22x9x75x6xf32>) -> tensor<22x9x75x6xf32>
    %8 = tosa.reduce_sum %7 {axis = 0 : i32} : (tensor<22x9x75x6xf32>) -> tensor<1x9x75x6xf32>
    %9 = tosa.pow %8, %8 : (tensor<1x9x75x6xf32>, tensor<1x9x75x6xf32>) -> tensor<1x9x75x6xf32>
    %10 = tosa.logical_xor %6, %6 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %11 = tosa.abs %7 : (tensor<22x9x75x6xf32>) -> tensor<22x9x75x6xf32>
    %12 = tosa.clz %5 : (tensor<37x92x570x18xi16>) -> tensor<37x92x570x18xi16>
    %13 = tosa.add %7, %11 : (tensor<22x9x75x6xf32>, tensor<22x9x75x6xf32>) -> tensor<22x9x75x6xf32>
    return %9, %10, %12, %13 : tensor<1x9x75x6xf32>, tensor<i1>, tensor<37x92x570x18xi16>, tensor<22x9x75x6xf32>
  }
}
