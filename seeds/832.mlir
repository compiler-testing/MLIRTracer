module {
  func.func @main(%arg0: tensor<77x92x11xi32>, %arg1: tensor<69x62x75x64x86xf32>) -> (tensor<154x184x1xi1>, tensor<2x6x7x8x1xf32>, tensor<154x184x1xi1>, tensor<77x92x11xi32>) {
    %0 = tosa.abs %arg0 : (tensor<77x92x11xi32>) -> tensor<77x92x11xi32>
    %1 = tosa.clz %0 : (tensor<77x92x11xi32>) -> tensor<77x92x11xi32>
    %2 = tosa.bitwise_or %1, %0 : (tensor<77x92x11xi32>, tensor<77x92x11xi32>) -> tensor<77x92x11xi32>
    %t_3 = tosa.const_shape {values = dense<[ 2, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %3 = tosa.tile %2, %t_3 : (tensor<77x92x11xi32>, !tosa.shape<3>) -> tensor<154x184x11xi32>
    %4 = tosa.clamp %3 {min_val = 31 : i32, max_val = 120 : i32} : (tensor<154x184x11xi32>) -> tensor<154x184x11xi32>
    %5 = tosa.concat %4, %4 {axis = 2 : i32} : (tensor<154x184x11xi32>, tensor<154x184x11xi32>) -> tensor<154x184x22xi32>
    %6 = tosa.floor %arg1 : (tensor<69x62x75x64x86xf32>) -> tensor<69x62x75x64x86xf32>
    %7 = tosa.equal %5, %5 : (tensor<154x184x22xi32>, tensor<154x184x22xi32>) -> tensor<154x184x22xi1>
    %8 = tosa.reduce_min %7 {axis = 2 : i32} : (tensor<154x184x22xi1>) -> tensor<154x184x1xi1>
    %s_9_start = tosa.const_shape {values = dense<[ 39, 56, 63, 56, 14 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_9_size = tosa.const_shape {values = dense<[ 2, 6, 7, 8, 1 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %9 = tosa.slice %6, %s_9_start, %s_9_size : (tensor<69x62x75x64x86xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<2x6x7x8x1xf32>
    %10 = tosa.clz %8 : (tensor<154x184x1xi1>) -> tensor<154x184x1xi1>
    %11 = tosa.add %9, %9 : (tensor<2x6x7x8x1xf32>, tensor<2x6x7x8x1xf32>) -> tensor<2x6x7x8x1xf32>
    %12 = tosa.reverse %10 {axis = 0 : i32} : (tensor<154x184x1xi1>) -> tensor<154x184x1xi1>
    %13 = tosa.logical_or %12, %8 : (tensor<154x184x1xi1>, tensor<154x184x1xi1>) -> tensor<154x184x1xi1>
    %14 = tosa.minimum %11, %11 : (tensor<2x6x7x8x1xf32>, tensor<2x6x7x8x1xf32>) -> tensor<2x6x7x8x1xf32>
    %15 = tosa.logical_and %8, %8 : (tensor<154x184x1xi1>, tensor<154x184x1xi1>) -> tensor<154x184x1xi1>
    %16 = tosa.logical_right_shift %0, %2 : (tensor<77x92x11xi32>, tensor<77x92x11xi32>) -> tensor<77x92x11xi32>
    return %13, %14, %15, %16 : tensor<154x184x1xi1>, tensor<2x6x7x8x1xf32>, tensor<154x184x1xi1>, tensor<77x92x11xi32>
  }
}
