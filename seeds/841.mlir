module {
  func.func @main(%arg0: tensor<89x9x1xi32>, %arg1: tensor<1x1x1xi32>, %arg2: tensor<15x26x37xi1>) -> (tensor<11x1x9xi1>, tensor<89x9x1xi32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<89x9x1xi32>, tensor<1x1x1xi32>) -> tensor<89x9x1xi32>
    %1 = tosa.clamp %0 {min_val = 11 : i32, max_val = 65 : i32} : (tensor<89x9x1xi32>) -> tensor<89x9x1xi32>
    %2 = tosa.bitwise_not %1 : (tensor<89x9x1xi32>) -> tensor<89x9x1xi32>
    %3 = tosa.clamp %2 {min_val = 11 : i32, max_val = 65 : i32} : (tensor<89x9x1xi32>) -> tensor<89x9x1xi32>
    %4 = tosa.reduce_all %arg2 {axis = 2 : i32} : (tensor<15x26x37xi1>) -> tensor<15x26x1xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 4, 13, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_5_size = tosa.const_shape {values = dense<[ 11, 9, 9 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<15x26x1xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<11x9x9xi1>
    %6 = tosa.reduce_product %5 {axis = 1 : i32} : (tensor<11x9x9xi1>) -> tensor<11x1x9xi1>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = tosa.negate %3, %in_zp_7, %out_zp_7 : (tensor<89x9x1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<89x9x1xi32>
    return %6, %7 : tensor<11x1x9xi1>, tensor<89x9x1xi32>
  }
}
