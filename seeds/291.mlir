module {
  func.func @main(%arg0: tensor<11x84x78xi64>, %arg1: tensor<72x62x80xf32>, %arg2: tensor<59x57x81x51x67xi1>, %arg3: tensor<1x57x1x1x1xi1>) -> (tensor<72x62x80xf32>, tensor<357120xf32>, tensor<357120xf32>, tensor<357120xf32>, tensor<864864xi64>, tensor<i32>, tensor<357120xf32>, tensor<59x57x81x51x67xi1>, tensor<72x62x1xf32>, tensor<1xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 2, 3, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = tosa.tile %arg0, %t_0 : (tensor<11x84x78xi64>, !tosa.shape<3>) -> tensor<22x252x156xi64>
    %r_1 = tosa.const_shape {values = dense<[ 864864 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %1 = tosa.reshape %0, %r_1 : (tensor<22x252x156xi64>, !tosa.shape<1>) -> tensor<864864xi64>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<864864xi64>, tensor<864864xi64>) -> tensor<864864xi64>
    %3 = tosa.clz %2 : (tensor<864864xi64>) -> tensor<864864xi64>
    %4 = tosa.tanh %arg1 : (tensor<72x62x80xf32>) -> tensor<72x62x80xf32>
    %5 = tosa.bitwise_and %3, %1 : (tensor<864864xi64>, tensor<864864xi64>) -> tensor<864864xi64>
    %6 = tosa.concat %5, %5 {axis = 0 : i32} : (tensor<864864xi64>, tensor<864864xi64>) -> tensor<1729728xi64>
    %7 = tosa.add %6, %6 : (tensor<1729728xi64>, tensor<1729728xi64>) -> tensor<1729728xi64>
    %8 = tosa.argmax %7 {axis = 0 : i32} : (tensor<1729728xi64>) -> tensor<i32>
    %9 = tosa.concat %4, %4 {axis = 2 : i32} : (tensor<72x62x80xf32>, tensor<72x62x80xf32>) -> tensor<72x62x160xf32>
    %10 = tosa.logical_or %arg2, %arg3 : (tensor<59x57x81x51x67xi1>, tensor<1x57x1x1x1xi1>) -> tensor<59x57x81x51x67xi1>
    %11 = tosa.sub %4, %4 : (tensor<72x62x80xf32>, tensor<72x62x80xf32>) -> tensor<72x62x80xf32>
    %r_12 = tosa.const_shape {values = dense<[ 357120 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %12 = tosa.reshape %4, %r_12 : (tensor<72x62x80xf32>, !tosa.shape<1>) -> tensor<357120xf32>
    %13 = tosa.abs %12 : (tensor<357120xf32>) -> tensor<357120xf32>
    %14 = tosa.exp %12 : (tensor<357120xf32>) -> tensor<357120xf32>
    %15 = tosa.intdiv %8, %8 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %16 = tosa.clamp %12 {min_val = -3.000000e+01 : f32, max_val = 1.100000e+02 : f32} : (tensor<357120xf32>) -> tensor<357120xf32>
    %17 = tosa.reduce_min %9 {axis = 2 : i32} : (tensor<72x62x160xf32>) -> tensor<72x62x1xf32>
    %18 = tosa.exp %17 : (tensor<72x62x1xf32>) -> tensor<72x62x1xf32>
    %19 = tosa.logical_xor %10, %10 : (tensor<59x57x81x51x67xi1>, tensor<59x57x81x51x67xi1>) -> tensor<59x57x81x51x67xi1>
    %20 = tosa.logical_left_shift %5, %2 : (tensor<864864xi64>, tensor<864864xi64>) -> tensor<864864xi64>
    %21 = tosa.bitwise_not %15 : (tensor<i32>) -> tensor<i32>
    %r_22 = tosa.const_shape {values = dense<[ 357120 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %22 = tosa.reshape %12, %r_22 : (tensor<357120xf32>, !tosa.shape<1>) -> tensor<357120xf32>
    %23 = tosa.reduce_min %12 {axis = 0 : i32} : (tensor<357120xf32>) -> tensor<1xf32>
    %24 = tosa.bitwise_xor %19, %10 : (tensor<59x57x81x51x67xi1>, tensor<59x57x81x51x67xi1>) -> tensor<59x57x81x51x67xi1>
    %25 = tosa.tanh %18 : (tensor<72x62x1xf32>) -> tensor<72x62x1xf32>
    %26 = tosa.reduce_min %23 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    return %11, %13, %14, %16, %20, %21, %22, %24, %25, %26 : tensor<72x62x80xf32>, tensor<357120xf32>, tensor<357120xf32>, tensor<357120xf32>, tensor<864864xi64>, tensor<i32>, tensor<357120xf32>, tensor<59x57x81x51x67xi1>, tensor<72x62x1xf32>, tensor<1xf32>
  }
}
