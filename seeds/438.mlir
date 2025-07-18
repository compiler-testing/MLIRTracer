module {
  func.func @main(%arg0: tensor<72x100x47x55xi16>, %arg1: tensor<47x56xi1>, %arg2: tensor<12x82x92x39xf32>) -> (tensor<2632xi1>, tensor<1x100x47x55xi16>, tensor<82x92x39xi32>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<72x100x47x55xi16>) -> tensor<1x100x47x55xi16>
    %1 = tosa.logical_not %arg1 : (tensor<47x56xi1>) -> tensor<47x56xi1>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<1x100x47x55xi16>, tensor<1x100x47x55xi16>) -> tensor<1x100x47x55xi16>
    %r_3 = tosa.const_shape {values = dense<[ 2632 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.reshape %1, %r_3 : (tensor<47x56xi1>, !tosa.shape<1>) -> tensor<2632xi1>
    %4 = tosa.sub %3, %3 : (tensor<2632xi1>, tensor<2632xi1>) -> tensor<2632xi1>
    %5 = tosa.rsqrt %arg2 : (tensor<12x82x92x39xf32>) -> tensor<12x82x92x39xf32>
    %6 = tosa.bitwise_and %2, %2 : (tensor<1x100x47x55xi16>, tensor<1x100x47x55xi16>) -> tensor<1x100x47x55xi16>
    %7 = tosa.argmax %5 {axis = 0 : i32} : (tensor<12x82x92x39xf32>) -> tensor<82x92x39xi32>
    return %4, %6, %7 : tensor<2632xi1>, tensor<1x100x47x55xi16>, tensor<82x92x39xi32>
  }
}
