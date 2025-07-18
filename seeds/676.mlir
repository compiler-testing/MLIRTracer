module {
  func.func @main(%arg0: tensor<41x66xi32>, %arg1: tensor<63x25x32x55x50x67xf32>, %arg2: tensor<49x35x56x52xi1>) -> (tensor<3x2x1x738xi32>, tensor<1x1x11x246xi32>, tensor<63x25x32x55x50x67xf32>, tensor<49x35x56x1xi1>, tensor<3x2x11x738xi32>, tensor<32x50x67x63x55x25xf32>, tensor<32x50x67x63x55x25xf32>) {
    %r_0 = tosa.const_shape {values = dense<[ 1, 1, 11, 246 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<41x66xi32>, !tosa.shape<4>) -> tensor<1x1x11x246xi32>
    %1 = tosa.floor %arg1 : (tensor<63x25x32x55x50x67xf32>) -> tensor<63x25x32x55x50x67xf32>
    %2 = tosa.tanh %1 : (tensor<63x25x32x55x50x67xf32>) -> tensor<63x25x32x55x50x67xf32>
    %in_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_3 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<63x25x32x55x50x67xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<63x25x32x55x50x67xf32>
    %t_4 = tosa.const_shape {values = dense<[ 3, 2, 1, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.tile %0, %t_4 : (tensor<1x1x11x246xi32>, !tosa.shape<4>) -> tensor<3x2x11x738xi32>
    %5 = tosa.reduce_product %4 {axis = 2 : i32} : (tensor<3x2x11x738xi32>) -> tensor<3x2x1x738xi32>
    %6 = tosa.logical_left_shift %0, %0 : (tensor<1x1x11x246xi32>, tensor<1x1x11x246xi32>) -> tensor<1x1x11x246xi32>
    %7 = "tosa.const"() {values = dense<[2, 4, 5, 0, 3, 1]> : tensor<6xi32>} : () -> tensor<6xi32>
    %8 = tosa.transpose %1 {perms = array<i32: 2, 4, 5, 0, 3, 1>} : (tensor<63x25x32x55x50x67xf32>) -> tensor<32x50x67x63x55x25xf32>
    %9 = tosa.intdiv %6, %6 : (tensor<1x1x11x246xi32>, tensor<1x1x11x246xi32>) -> tensor<1x1x11x246xi32>
    %10 = tosa.intdiv %4, %4 : (tensor<3x2x11x738xi32>, tensor<3x2x11x738xi32>) -> tensor<3x2x11x738xi32>
    %11 = tosa.abs %3 : (tensor<63x25x32x55x50x67xf32>) -> tensor<63x25x32x55x50x67xf32>
    %12 = tosa.maximum %11, %11 : (tensor<63x25x32x55x50x67xf32>, tensor<63x25x32x55x50x67xf32>) -> tensor<63x25x32x55x50x67xf32>
    %13 = tosa.reduce_any %arg2 {axis = 3 : i32} : (tensor<49x35x56x52xi1>) -> tensor<49x35x56x1xi1>
    %14 = tosa.bitwise_and %13, %13 : (tensor<49x35x56x1xi1>, tensor<49x35x56x1xi1>) -> tensor<49x35x56x1xi1>
    %15 = tosa.intdiv %4, %10 : (tensor<3x2x11x738xi32>, tensor<3x2x11x738xi32>) -> tensor<3x2x11x738xi32>
    %16 = tosa.rsqrt %8 : (tensor<32x50x67x63x55x25xf32>) -> tensor<32x50x67x63x55x25xf32>
    %17 = tosa.floor %8 : (tensor<32x50x67x63x55x25xf32>) -> tensor<32x50x67x63x55x25xf32>
    return %5, %9, %12, %14, %15, %16, %17 : tensor<3x2x1x738xi32>, tensor<1x1x11x246xi32>, tensor<63x25x32x55x50x67xf32>, tensor<49x35x56x1xi1>, tensor<3x2x11x738xi32>, tensor<32x50x67x63x55x25xf32>, tensor<32x50x67x63x55x25xf32>
  }
}
