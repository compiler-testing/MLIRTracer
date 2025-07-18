module {
  func.func @main(%arg0: tensor<18x54x96xi8>, %arg1: tensor<18x1x96xi8>, %arg2: tensor<93xf32>) -> (tensor<8x3xi32>, tensor<93xf32>, tensor<1xf32>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<18x54x96xi8>, tensor<18x1x96xi8>) -> tensor<18x54x96xi1>
    %1 = tosa.logical_and %0, %0 : (tensor<18x54x96xi1>, tensor<18x54x96xi1>) -> tensor<18x54x96xi1>
    %s_2_start = tosa.const_shape {values = dense<[ 9, 18, 17 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_2_size = tosa.const_shape {values = dense<[ 9, 8, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<18x54x96xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<9x8x3xi1>
    %3 = tosa.clz %2 : (tensor<9x8x3xi1>) -> tensor<9x8x3xi1>
    %4 = tosa.reduce_max %3 {axis = 0 : i32} : (tensor<9x8x3xi1>) -> tensor<1x8x3xi1>
    %5 = tosa.argmax %4 {axis = 0 : i32} : (tensor<1x8x3xi1>) -> tensor<8x3xi32>
    %6 = tosa.sub %5, %5 : (tensor<8x3xi32>, tensor<8x3xi32>) -> tensor<8x3xi32>
    %7 = tosa.logical_right_shift %6, %5 : (tensor<8x3xi32>, tensor<8x3xi32>) -> tensor<8x3xi32>
    %8 = tosa.reverse %7 {axis = 0 : i32} : (tensor<8x3xi32>) -> tensor<8x3xi32>
    %9 = tosa.logical_left_shift %8, %8 : (tensor<8x3xi32>, tensor<8x3xi32>) -> tensor<8x3xi32>
    %10 = tosa.maximum %9, %8 : (tensor<8x3xi32>, tensor<8x3xi32>) -> tensor<8x3xi32>
    %11 = tosa.minimum %10, %9 : (tensor<8x3xi32>, tensor<8x3xi32>) -> tensor<8x3xi32>
    %12 = tosa.log %arg2 : (tensor<93xf32>) -> tensor<93xf32>
    %13 = tosa.rsqrt %12 : (tensor<93xf32>) -> tensor<93xf32>
    %14 = tosa.reduce_product %12 {axis = 0 : i32} : (tensor<93xf32>) -> tensor<1xf32>
    return %11, %13, %14 : tensor<8x3xi32>, tensor<93xf32>, tensor<1xf32>
  }
}
