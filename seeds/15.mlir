module {
  func.func @main(%arg0: tensor<98x11xf32>, %arg1: tensor<5x4xi1>, %arg2: tensor<1x4xi1>) -> (tensor<5x4xi1>, tensor<12x11xi1>, tensor<98x11xf32>, tensor<294x33xi1>, tensor<294x33xf32>) {
    %0 = tosa.tanh %arg0 : (tensor<98x11xf32>) -> tensor<98x11xf32>
    %t_1 = tosa.const_shape {values = dense<[ 3, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %0, %t_1 : (tensor<98x11xf32>, !tosa.shape<2>) -> tensor<294x33xf32>
    %2 = tosa.identity %1 : (tensor<294x33xf32>) -> tensor<294x33xf32>
    %3 = tosa.clamp %2 {min_val = 3.600000e+01 : f32, max_val = 4.000000e+01 : f32} : (tensor<294x33xf32>) -> tensor<294x33xf32>
    %4 = tosa.logical_xor %arg1, %arg2 : (tensor<5x4xi1>, tensor<1x4xi1>) -> tensor<5x4xi1>
    %5 = tosa.abs %3 : (tensor<294x33xf32>) -> tensor<294x33xf32>
    %6 = tosa.equal %5, %5 : (tensor<294x33xf32>, tensor<294x33xf32>) -> tensor<294x33xi1>
    %7 = tosa.add %4, %4 : (tensor<5x4xi1>, tensor<5x4xi1>) -> tensor<5x4xi1>
    %8 = tosa.arithmetic_right_shift %7, %4 {round = true} : (tensor<5x4xi1>, tensor<5x4xi1>) -> tensor<5x4xi1>
    %s_9_start = tosa.const_shape {values = dense<[ 36, 22 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_9_size = tosa.const_shape {values = dense<[ 12, 11 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %9 = tosa.slice %6, %s_9_start, %s_9_size : (tensor<294x33xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<12x11xi1>
    %10 = tosa.logical_or %8, %8 : (tensor<5x4xi1>, tensor<5x4xi1>) -> tensor<5x4xi1>
    %11 = tosa.bitwise_or %9, %9 : (tensor<12x11xi1>, tensor<12x11xi1>) -> tensor<12x11xi1>
    %12 = tosa.tanh %0 : (tensor<98x11xf32>) -> tensor<98x11xf32>
    %13 = tosa.equal %1, %5 : (tensor<294x33xf32>, tensor<294x33xf32>) -> tensor<294x33xi1>
    %14 = tosa.pow %2, %2 : (tensor<294x33xf32>, tensor<294x33xf32>) -> tensor<294x33xf32>
    return %10, %11, %12, %13, %14 : tensor<5x4xi1>, tensor<12x11xi1>, tensor<98x11xf32>, tensor<294x33xi1>, tensor<294x33xf32>
  }
}
