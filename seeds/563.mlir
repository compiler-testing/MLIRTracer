module {
  func.func @main(%arg0: tensor<31x59x24x71x25x39xi16>, %arg1: tensor<1x59x1x71x1x1xi16>, %arg2: tensor<82x57x47x18x83xf32>, %arg3: tensor<50x13x48x56xf32>, %arg4: tensor<78x30x39x11xf32>, %arg5: tensor<78xf32>, %arg6: tensor<27x18x86x8x51x28xi32>, %arg7: tensor<27x18x1x8x51x28xi32>, %arg8: tensor<i1>, %arg9: tensor<i1>) -> (tensor<31x59x24x71x25x39xi16>, tensor<82x57x47x18x83xf32>, tensor<50x44x89x1xf32>, tensor<4x9x6x10xf32>, tensor<1x44x89x78xf32>, tensor<27x18x86x8x51x28xi32>, tensor<i1>, tensor<50x44x89x78xf32>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<31x59x24x71x25x39xi16>, tensor<1x59x1x71x1x1xi16>) -> tensor<31x59x24x71x25x39xi16>
    %1 = tosa.reciprocal %arg2 : (tensor<82x57x47x18x83xf32>) -> tensor<82x57x47x18x83xf32>
    %input_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.transpose_conv2d %arg3, %arg4, %arg5, %input_zp_2, %weight_zp_2 {acc_type = f32, out_pad = array<i64: 1, 1, 2, 1>, stride = array<i64: 1, 1>, out_shape = array<i64: 50, 44, 89, 78>} : (tensor<50x13x48x56xf32>, tensor<78x30x39x11xf32>, tensor<78xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<50x44x89x78xf32>
    %3 = tosa.reduce_min %2 {axis = 3 : i32} : (tensor<50x44x89x78xf32>) -> tensor<50x44x89x1xf32>
    %s_4_start = tosa.const_shape {values = dense<[ 32, 14, 36, 21 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_4_size = tosa.const_shape {values = dense<[ 4, 9, 6, 10 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.slice %2, %s_4_start, %s_4_size : (tensor<50x44x89x78xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<4x9x6x10xf32>
    %5 = tosa.exp %2 : (tensor<50x44x89x78xf32>) -> tensor<50x44x89x78xf32>
    %6 = tosa.intdiv %arg6, %arg7 : (tensor<27x18x86x8x51x28xi32>, tensor<27x18x1x8x51x28xi32>) -> tensor<27x18x86x8x51x28xi32>
    %7 = tosa.log %5 : (tensor<50x44x89x78xf32>) -> tensor<50x44x89x78xf32>
    %8 = tosa.reduce_min %7 {axis = 0 : i32} : (tensor<50x44x89x78xf32>) -> tensor<1x44x89x78xf32>
    %9 = tosa.logical_or %arg8, %arg9 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %10 = tosa.bitwise_and %6, %6 : (tensor<27x18x86x8x51x28xi32>, tensor<27x18x86x8x51x28xi32>) -> tensor<27x18x86x8x51x28xi32>
    %11 = tosa.logical_not %9 : (tensor<i1>) -> tensor<i1>
    %12 = tosa.exp %7 : (tensor<50x44x89x78xf32>) -> tensor<50x44x89x78xf32>
    return %0, %1, %3, %4, %8, %10, %11, %12 : tensor<31x59x24x71x25x39xi16>, tensor<82x57x47x18x83xf32>, tensor<50x44x89x1xf32>, tensor<4x9x6x10xf32>, tensor<1x44x89x78xf32>, tensor<27x18x86x8x51x28xi32>, tensor<i1>, tensor<50x44x89x78xf32>
  }
}
