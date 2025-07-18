module {
  func.func @main(%arg0: tensor<59x12x86x14xi16>, %arg1: tensor<88x70x69xf32>) -> (tensor<1x1x258x42xi16>, tensor<88x70x69xf32>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<59x12x86x14xi16>) -> tensor<1x12x86x14xi16>
    %t_1 = tosa.const_shape {values = dense<[ 1, 3, 3, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.tile %0, %t_1 : (tensor<1x12x86x14xi16>, !tosa.shape<4>) -> tensor<1x36x258x42xi16>
    %2 = tosa.exp %arg1 : (tensor<88x70x69xf32>) -> tensor<88x70x69xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<1x36x258x42xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<1x36x258x42xi16>
    %4 = tosa.arithmetic_right_shift %3, %1 {round = true} : (tensor<1x36x258x42xi16>, tensor<1x36x258x42xi16>) -> tensor<1x36x258x42xi16>
    %5 = tosa.reduce_min %4 {axis = 1 : i32} : (tensor<1x36x258x42xi16>) -> tensor<1x1x258x42xi16>
    %6 = tosa.sigmoid %2 : (tensor<88x70x69xf32>) -> tensor<88x70x69xf32>
    %7 = tosa.sub %6, %2 : (tensor<88x70x69xf32>, tensor<88x70x69xf32>) -> tensor<88x70x69xf32>
    return %5, %7 : tensor<1x1x258x42xi16>, tensor<88x70x69xf32>
  }
}
