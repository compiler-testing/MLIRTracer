module {
  func.func @main(%arg0: tensor<45x85x61x11x92xi16>, %arg1: tensor<24x26x3x11xi1>, %arg2: tensor<91x40x27x99x36x22xf32>) -> (tensor<45x85x61x11x92xi16>, tensor<91x40x27x99x36x22xf32>, tensor<72x78x9x22xi1>, tensor<1x26x3x11xi1>, tensor<1x26x3x11xi1>) {
    %in_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = tosa.negate %arg0, %in_zp_0, %out_zp_0 : (tensor<45x85x61x11x92xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<45x85x61x11x92xi16>
    %1 = tosa.reverse %arg1 {axis = 3 : i32} : (tensor<24x26x3x11xi1>) -> tensor<24x26x3x11xi1>
    %2 = tosa.bitwise_or %0, %0 : (tensor<45x85x61x11x92xi16>, tensor<45x85x61x11x92xi16>) -> tensor<45x85x61x11x92xi16>
    %3 = tosa.tanh %arg2 : (tensor<91x40x27x99x36x22xf32>) -> tensor<91x40x27x99x36x22xf32>
    %t_4 = tosa.const_shape {values = dense<[ 3, 3, 3, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.tile %1, %t_4 : (tensor<24x26x3x11xi1>, !tosa.shape<4>) -> tensor<72x78x9x22xi1>
    %5 = tosa.bitwise_xor %1, %1 : (tensor<24x26x3x11xi1>, tensor<24x26x3x11xi1>) -> tensor<24x26x3x11xi1>
    %6 = tosa.reduce_sum %5 {axis = 0 : i32} : (tensor<24x26x3x11xi1>) -> tensor<1x26x3x11xi1>
    %7 = tosa.add %4, %4 : (tensor<72x78x9x22xi1>, tensor<72x78x9x22xi1>) -> tensor<72x78x9x22xi1>
    %8 = tosa.reverse %7 {axis = 1 : i32} : (tensor<72x78x9x22xi1>) -> tensor<72x78x9x22xi1>
    %9 = tosa.sub %6, %6 : (tensor<1x26x3x11xi1>, tensor<1x26x3x11xi1>) -> tensor<1x26x3x11xi1>
    %10 = tosa.clz %6 : (tensor<1x26x3x11xi1>) -> tensor<1x26x3x11xi1>
    return %2, %3, %8, %9, %10 : tensor<45x85x61x11x92xi16>, tensor<91x40x27x99x36x22xf32>, tensor<72x78x9x22xi1>, tensor<1x26x3x11xi1>, tensor<1x26x3x11xi1>
  }
}
