module {
  func.func @main(%arg0: tensor<78xi8>, %arg1: tensor<81x90x86xf32>) -> (tensor<10xi1>, tensor<10xi1>, tensor<2x4xi32>, tensor<8x3x5xi1>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<78xi8>) -> tensor<78xi8>
    %1 = tosa.equal %0, %0 : (tensor<78xi8>, tensor<78xi8>) -> tensor<78xi1>
    %2 = tosa.reciprocal %arg1 : (tensor<81x90x86xf32>) -> tensor<81x90x86xf32>
    %3 = tosa.bitwise_or %1, %1 : (tensor<78xi1>, tensor<78xi1>) -> tensor<78xi1>
    %4 = tosa.logical_right_shift %3, %1 : (tensor<78xi1>, tensor<78xi1>) -> tensor<78xi1>
    %5 = tosa.clamp %2 {min_val = 2.300000e+01 : f32, max_val = 4.900000e+01 : f32} : (tensor<81x90x86xf32>) -> tensor<81x90x86xf32>
    %6 = tosa.logical_and %4, %1 : (tensor<78xi1>, tensor<78xi1>) -> tensor<78xi1>
    %s_7_start = tosa.const_shape {values = dense<[ 68 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_7_size = tosa.const_shape {values = dense<[ 10 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %7 = tosa.slice %6, %s_7_start, %s_7_size : (tensor<78xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<10xi1>
    %8 = tosa.logical_and %7, %7 : (tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
    %9 = tosa.bitwise_and %8, %7 : (tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
    %s_10_start = tosa.const_shape {values = dense<[ 45, 27, 18 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_10_size = tosa.const_shape {values = dense<[ 2, 2, 4 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %10 = tosa.slice %5, %s_10_start, %s_10_size : (tensor<81x90x86xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<2x2x4xf32>
    %11 = tosa.exp %10 : (tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %12 = tosa.bitwise_and %7, %7 : (tensor<10xi1>, tensor<10xi1>) -> tensor<10xi1>
    %13 = tosa.argmax %11 {axis = 0 : i32} : (tensor<2x2x4xf32>) -> tensor<2x4xi32>
    %14 = tosa.bitwise_and %13, %13 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
    %15 = tosa.clz %14 : (tensor<2x4xi32>) -> tensor<2x4xi32>
    %s_16_start = tosa.const_shape {values = dense<[ 0, 0, 0 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_16_size = tosa.const_shape {values = dense<[ 8, 3, 5 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %16 = tosa.slice %10, %s_16_start, %s_16_size : (tensor<2x2x4xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<8x3x5xf32>
    %17 = tosa.floor %16 : (tensor<8x3x5xf32>) -> tensor<8x3x5xf32>
    %18 = tosa.greater_equal %17, %17 : (tensor<8x3x5xf32>, tensor<8x3x5xf32>) -> tensor<8x3x5xi1>
    return %9, %12, %15, %18 : tensor<10xi1>, tensor<10xi1>, tensor<2x4xi32>, tensor<8x3x5xi1>
  }
}
