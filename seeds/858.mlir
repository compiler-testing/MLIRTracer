module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<47x50x85x73xf32>, %arg3: tensor<55x89x62x59x44xi1>, %arg4: tensor<1x89x62x1x44xi1>) -> (tensor<i8>, tensor<55x89x62x59x44xi1>, tensor<47x50x73xi32>, tensor<55x89x62x59x44xi1>, tensor<47x50x85x73xi1>, tensor<47x50x85x73xf32>, tensor<5x2x11x4x3xi1>, tensor<47x1x85x73xf32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %1 = tosa.bitwise_or %0, %0 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %2 = tosa.tanh %arg2 : (tensor<47x50x85x73xf32>) -> tensor<47x50x85x73xf32>
    %3 = tosa.logical_xor %arg3, %arg4 : (tensor<55x89x62x59x44xi1>, tensor<1x89x62x1x44xi1>) -> tensor<55x89x62x59x44xi1>
    %4 = tosa.argmax %2 {axis = 2 : i32} : (tensor<47x50x85x73xf32>) -> tensor<47x50x73xi32>
    %5 = tosa.floor %2 : (tensor<47x50x85x73xf32>) -> tensor<47x50x85x73xf32>
    %6 = tosa.reduce_sum %5 {axis = 1 : i32} : (tensor<47x50x85x73xf32>) -> tensor<47x1x85x73xf32>
    %7 = tosa.arithmetic_right_shift %3, %3 {round = false} : (tensor<55x89x62x59x44xi1>, tensor<55x89x62x59x44xi1>) -> tensor<55x89x62x59x44xi1>
    %8 = tosa.bitwise_not %3 : (tensor<55x89x62x59x44xi1>) -> tensor<55x89x62x59x44xi1>
    %9 = tosa.clz %3 : (tensor<55x89x62x59x44xi1>) -> tensor<55x89x62x59x44xi1>
    %10 = tosa.logical_and %7, %8 : (tensor<55x89x62x59x44xi1>, tensor<55x89x62x59x44xi1>) -> tensor<55x89x62x59x44xi1>
    %11 = tosa.intdiv %4, %4 : (tensor<47x50x73xi32>, tensor<47x50x73xi32>) -> tensor<47x50x73xi32>
    %s_12_start = tosa.const_shape {values = dense<[ 36, 33, 32, 55, 20 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_12_size = tosa.const_shape {values = dense<[ 5, 2, 11, 4, 3 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %12 = tosa.slice %3, %s_12_start, %s_12_size : (tensor<55x89x62x59x44xi1>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<5x2x11x4x3xi1>
    %in_zp_13 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_13 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %13 = tosa.negate %10, %in_zp_13, %out_zp_13 : (tensor<55x89x62x59x44xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<55x89x62x59x44xi1>
    %14 = tosa.logical_and %12, %12 : (tensor<5x2x11x4x3xi1>, tensor<5x2x11x4x3xi1>) -> tensor<5x2x11x4x3xi1>
    %15 = tosa.floor %5 : (tensor<47x50x85x73xf32>) -> tensor<47x50x85x73xf32>
    %16 = tosa.sub %14, %14 : (tensor<5x2x11x4x3xi1>, tensor<5x2x11x4x3xi1>) -> tensor<5x2x11x4x3xi1>
    %17 = tosa.greater_equal %5, %15 : (tensor<47x50x85x73xf32>, tensor<47x50x85x73xf32>) -> tensor<47x50x85x73xi1>
    %18 = tosa.sigmoid %2 : (tensor<47x50x85x73xf32>) -> tensor<47x50x85x73xf32>
    %19 = tosa.arithmetic_right_shift %16, %14 {round = false} : (tensor<5x2x11x4x3xi1>, tensor<5x2x11x4x3xi1>) -> tensor<5x2x11x4x3xi1>
    %20 = tosa.exp %6 : (tensor<47x1x85x73xf32>) -> tensor<47x1x85x73xf32>
    return %1, %9, %11, %13, %17, %18, %19, %20 : tensor<i8>, tensor<55x89x62x59x44xi1>, tensor<47x50x73xi32>, tensor<55x89x62x59x44xi1>, tensor<47x50x85x73xi1>, tensor<47x50x85x73xf32>, tensor<5x2x11x4x3xi1>, tensor<47x1x85x73xf32>
  }
}
