module {
  func.func @main(%arg0: tensor<56x82x93x32x29x85xi1>, %arg1: tensor<7x35x62x68x33xi8>, %arg2: tensor<7x35x1x1x33xi8>, %arg3: tensor<12xi64>, %arg4: tensor<32xf32>, %arg5: tensor<71x5xi1>) -> (tensor<112x82x93x32x29x85xi1>, tensor<1xi64>, tensor<32xf32>, tensor<1x5xi1>, tensor<31x28x7854x5xi8>, tensor<4x12x11x12x9xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<56x82x93x32x29x85xi1>) -> tensor<56x82x93x32x29x85xi1>
    %1 = tosa.logical_not %0 : (tensor<56x82x93x32x29x85xi1>) -> tensor<56x82x93x32x29x85xi1>
    %2 = tosa.minimum %arg1, %arg2 : (tensor<7x35x62x68x33xi8>, tensor<7x35x1x1x33xi8>) -> tensor<7x35x62x68x33xi8>
    %3 = tosa.bitwise_and %1, %1 : (tensor<56x82x93x32x29x85xi1>, tensor<56x82x93x32x29x85xi1>) -> tensor<56x82x93x32x29x85xi1>
    %4 = tosa.concat %3, %0 {axis = 0 : i32} : (tensor<56x82x93x32x29x85xi1>, tensor<56x82x93x32x29x85xi1>) -> tensor<112x82x93x32x29x85xi1>
    %5 = tosa.logical_left_shift %2, %2 : (tensor<7x35x62x68x33xi8>, tensor<7x35x62x68x33xi8>) -> tensor<7x35x62x68x33xi8>
    %6 = tosa.reduce_max %arg3 {axis = 0 : i32} : (tensor<12xi64>) -> tensor<1xi64>
    %r_7 = tosa.const_shape {values = dense<[ 31, 28, 7854, 5 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %7 = tosa.reshape %5, %r_7 : (tensor<7x35x62x68x33xi8>, !tosa.shape<4>) -> tensor<31x28x7854x5xi8>
    %8 = tosa.maximum %7, %7 : (tensor<31x28x7854x5xi8>, tensor<31x28x7854x5xi8>) -> tensor<31x28x7854x5xi8>
    %s_9_start = tosa.const_shape {values = dense<[ 2, 6, 4, 2, 7 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_9_size = tosa.const_shape {values = dense<[ 4, 12, 11, 12, 9 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %9 = tosa.slice %5, %s_9_start, %s_9_size : (tensor<7x35x62x68x33xi8>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<4x12x11x12x9xi8>
    %10 = tosa.maximum %8, %7 : (tensor<31x28x7854x5xi8>, tensor<31x28x7854x5xi8>) -> tensor<31x28x7854x5xi8>
    %11 = tosa.equal %9, %9 : (tensor<4x12x11x12x9xi8>, tensor<4x12x11x12x9xi8>) -> tensor<4x12x11x12x9xi1>
    %12 = tosa.ceil %arg4 : (tensor<32xf32>) -> tensor<32xf32>
    %13 = tosa.reduce_all %arg5 {axis = 0 : i32} : (tensor<71x5xi1>) -> tensor<1x5xi1>
    %14 = tosa.bitwise_xor %10, %8 : (tensor<31x28x7854x5xi8>, tensor<31x28x7854x5xi8>) -> tensor<31x28x7854x5xi8>
    %15 = tosa.bitwise_and %11, %11 : (tensor<4x12x11x12x9xi1>, tensor<4x12x11x12x9xi1>) -> tensor<4x12x11x12x9xi1>
    return %4, %6, %12, %13, %14, %15 : tensor<112x82x93x32x29x85xi1>, tensor<1xi64>, tensor<32xf32>, tensor<1x5xi1>, tensor<31x28x7854x5xi8>, tensor<4x12x11x12x9xi1>
  }
}
