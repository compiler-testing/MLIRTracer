module {
  func.func @main(%arg0: tensor<6x11x7x78xi64>, %arg1: tensor<1x1x1x1xi64>, %arg2: tensor<83x2x46xi8>, %arg3: tensor<83x2x1xi8>, %arg4: tensor<62x8x61x57x36x25xf32>) -> (tensor<6x11x7x78xi1>, tensor<62x8x61x57x36x25xf32>, tensor<4x6x12xi8>, tensor<83x1x46xi8>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<6x11x7x78xi64>, tensor<1x1x1x1xi64>) -> tensor<6x11x7x78xi1>
    %1 = tosa.logical_xor %0, %0 : (tensor<6x11x7x78xi1>, tensor<6x11x7x78xi1>) -> tensor<6x11x7x78xi1>
    %2 = tosa.maximum %arg2, %arg3 : (tensor<83x2x46xi8>, tensor<83x2x1xi8>) -> tensor<83x2x46xi8>
    %3 = tosa.bitwise_xor %2, %2 : (tensor<83x2x46xi8>, tensor<83x2x46xi8>) -> tensor<83x2x46xi8>
    %4 = tosa.rsqrt %arg4 : (tensor<62x8x61x57x36x25xf32>) -> tensor<62x8x61x57x36x25xf32>
    %5 = tosa.bitwise_xor %1, %0 : (tensor<6x11x7x78xi1>, tensor<6x11x7x78xi1>) -> tensor<6x11x7x78xi1>
    %6 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<83x2x46xi8>) -> tensor<83x1x46xi8>
    %7 = tosa.reverse %6 {axis = 2 : i32} : (tensor<83x1x46xi8>) -> tensor<83x1x46xi8>
    %8 = tosa.reciprocal %4 : (tensor<62x8x61x57x36x25xf32>) -> tensor<62x8x61x57x36x25xf32>
    %s_9_start = tosa.const_shape {values = dense<[ 44, 0, 34 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_9_size = tosa.const_shape {values = dense<[ 4, 6, 12 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %9 = tosa.slice %7, %s_9_start, %s_9_size : (tensor<83x1x46xi8>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x6x12xi8>
    %10 = tosa.add %7, %6 : (tensor<83x1x46xi8>, tensor<83x1x46xi8>) -> tensor<83x1x46xi8>
    return %5, %8, %9, %10 : tensor<6x11x7x78xi1>, tensor<62x8x61x57x36x25xf32>, tensor<4x6x12xi8>, tensor<83x1x46xi8>
  }
}
