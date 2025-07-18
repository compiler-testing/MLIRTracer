module {
  func.func @main(%arg0: tensor<84x80x60x49x53xf32>, %arg1: tensor<63xi1>, %arg2: tensor<80x79x65x23xi32>, %arg3: tensor<1x79x1x1xi32>) -> (tensor<84x80x60x49x106xf32>, tensor<84x80x60x49x106xf32>, tensor<84x80x60x49x106xf32>, tensor<2xi1>, tensor<84x80x60x49x106xf32>, tensor<1xi1>, tensor<80x79x65x23xi1>) {
    %0 = tosa.tanh %arg0 : (tensor<84x80x60x49x53xf32>) -> tensor<84x80x60x49x53xf32>
    %1 = tosa.concat %0, %0 {axis = 4 : i32} : (tensor<84x80x60x49x53xf32>, tensor<84x80x60x49x53xf32>) -> tensor<84x80x60x49x106xf32>
    %2 = tosa.add %1, %1 : (tensor<84x80x60x49x106xf32>, tensor<84x80x60x49x106xf32>) -> tensor<84x80x60x49x106xf32>
    %3 = tosa.rsqrt %2 : (tensor<84x80x60x49x106xf32>) -> tensor<84x80x60x49x106xf32>
    %4 = tosa.pow %3, %1 : (tensor<84x80x60x49x106xf32>, tensor<84x80x60x49x106xf32>) -> tensor<84x80x60x49x106xf32>
    %in_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_5 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %5 = tosa.negate %4, %in_zp_5, %out_zp_5 : (tensor<84x80x60x49x106xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<84x80x60x49x106xf32>
    %6 = tosa.reciprocal %5 : (tensor<84x80x60x49x106xf32>) -> tensor<84x80x60x49x106xf32>
    %7 = tosa.ceil %6 : (tensor<84x80x60x49x106xf32>) -> tensor<84x80x60x49x106xf32>
    %8 = tosa.floor %7 : (tensor<84x80x60x49x106xf32>) -> tensor<84x80x60x49x106xf32>
    %9 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<63xi1>) -> tensor<1xi1>
    %10 = tosa.logical_left_shift %9, %9 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %11 = tosa.pow %8, %5 : (tensor<84x80x60x49x106xf32>, tensor<84x80x60x49x106xf32>) -> tensor<84x80x60x49x106xf32>
    %12 = tosa.clz %9 : (tensor<1xi1>) -> tensor<1xi1>
    %13 = tosa.pow %5, %5 : (tensor<84x80x60x49x106xf32>, tensor<84x80x60x49x106xf32>) -> tensor<84x80x60x49x106xf32>
    %14 = tosa.logical_xor %12, %10 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %15 = tosa.maximum %11, %4 : (tensor<84x80x60x49x106xf32>, tensor<84x80x60x49x106xf32>) -> tensor<84x80x60x49x106xf32>
    %16 = tosa.maximum %6, %6 : (tensor<84x80x60x49x106xf32>, tensor<84x80x60x49x106xf32>) -> tensor<84x80x60x49x106xf32>
    %s_17_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_17_size = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %17 = tosa.slice %10, %s_17_start, %s_17_size : (tensor<1xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<2xi1>
    %18 = tosa.arithmetic_right_shift %17, %17 {round = true} : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
    %19 = tosa.exp %1 : (tensor<84x80x60x49x106xf32>) -> tensor<84x80x60x49x106xf32>
    %20 = tosa.reduce_min %14 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %21 = tosa.intdiv %arg2, %arg3 : (tensor<80x79x65x23xi32>, tensor<1x79x1x1xi32>) -> tensor<80x79x65x23xi32>
    %22 = tosa.equal %21, %21 : (tensor<80x79x65x23xi32>, tensor<80x79x65x23xi32>) -> tensor<80x79x65x23xi1>
    return %13, %15, %16, %18, %19, %20, %22 : tensor<84x80x60x49x106xf32>, tensor<84x80x60x49x106xf32>, tensor<84x80x60x49x106xf32>, tensor<2xi1>, tensor<84x80x60x49x106xf32>, tensor<1xi1>, tensor<80x79x65x23xi1>
  }
}
