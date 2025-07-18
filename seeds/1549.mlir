module {
  func.func @main(%arg0: tensor<50x22x29x32x38x71xf32>, %arg1: tensor<51x7x45x3x47xi8>, %arg2: tensor<51x7x1x3x47xi8>, %arg3: tensor<4xi32>, %arg4: tensor<1xi32>) -> (tensor<51x7x45x3x47xi8>, tensor<50x22x29x32x38x71xf32>, tensor<4xi32>) {
    %0 = tosa.tanh %arg0 : (tensor<50x22x29x32x38x71xf32>) -> tensor<50x22x29x32x38x71xf32>
    %1 = tosa.logical_left_shift %arg1, %arg2 : (tensor<51x7x45x3x47xi8>, tensor<51x7x1x3x47xi8>) -> tensor<51x7x45x3x47xi8>
    %2 = tosa.tanh %0 : (tensor<50x22x29x32x38x71xf32>) -> tensor<50x22x29x32x38x71xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = tosa.negate %1, %in_zp_3, %out_zp_3 : (tensor<51x7x45x3x47xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<51x7x45x3x47xi8>
    %4 = tosa.minimum %2, %2 : (tensor<50x22x29x32x38x71xf32>, tensor<50x22x29x32x38x71xf32>) -> tensor<50x22x29x32x38x71xf32>
    %5 = tosa.intdiv %arg3, %arg4 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
    return %3, %4, %5 : tensor<51x7x45x3x47xi8>, tensor<50x22x29x32x38x71xf32>, tensor<4xi32>
  }
}
