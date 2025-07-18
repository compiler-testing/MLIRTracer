module {
  func.func @main(%arg0: tensor<37x12x11x20x63xi1>, %arg1: tensor<37x1x11x20x1xi1>, %arg2: tensor<1x56x66xf32>) -> (tensor<8x7x8x1x11xi1>, tensor<1x56x66xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<37x12x11x20x63xi1>, tensor<37x1x11x20x1xi1>) -> tensor<37x12x11x20x63xi1>
    %s_1_start = tosa.const_shape {values = dense<[ 6, 5, 3, 19, 14 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_1_size = tosa.const_shape {values = dense<[ 8, 7, 8, 1, 11 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<37x12x11x20x63xi1>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<8x7x8x1x11xi1>
    %in_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_2 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<8x7x8x1x11xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<8x7x8x1x11xi1>
    %3 = tosa.sigmoid %arg2 : (tensor<1x56x66xf32>) -> tensor<1x56x66xf32>
    %4 = tosa.logical_and %2, %2 : (tensor<8x7x8x1x11xi1>, tensor<8x7x8x1x11xi1>) -> tensor<8x7x8x1x11xi1>
    %in_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5 = tosa.negate %3, %in_zp_5, %out_zp_5 : (tensor<1x56x66xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x56x66xf32>
    return %4, %5 : tensor<8x7x8x1x11xi1>, tensor<1x56x66xf32>
  }
}
