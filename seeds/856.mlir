module {
  func.func @main(%arg0: tensor<100x97x1x62x44x97xf32>, %arg1: tensor<88xi1>) -> (tensor<100x97x1x62x44x97xi1>, tensor<100x97x1x62x44x97xi1>, tensor<100x97x1x62x44x97xf32>, tensor<100x97x1x62x44x97xf32>, tensor<i32>, tensor<100x97x1x62x44x97xf32>, tensor<4x11x1x4xi1>, tensor<4x11x1x2xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<100x97x1x62x44x97xf32>) -> tensor<100x97x1x62x44x97xf32>
    %1 = tosa.tanh %0 : (tensor<100x97x1x62x44x97xf32>) -> tensor<100x97x1x62x44x97xf32>
    %2 = tosa.logical_not %arg1 : (tensor<88xi1>) -> tensor<88xi1>
    %3 = tosa.rsqrt %1 : (tensor<100x97x1x62x44x97xf32>) -> tensor<100x97x1x62x44x97xf32>
    %4 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<88xi1>) -> tensor<1xi1>
    %5 = tosa.reduce_max %4 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %6 = tosa.logical_right_shift %5, %5 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %7 = tosa.greater_equal %1, %1 : (tensor<100x97x1x62x44x97xf32>, tensor<100x97x1x62x44x97xf32>) -> tensor<100x97x1x62x44x97xi1>
    %8 = tosa.greater %3, %0 : (tensor<100x97x1x62x44x97xf32>, tensor<100x97x1x62x44x97xf32>) -> tensor<100x97x1x62x44x97xi1>
    %9 = tosa.rsqrt %1 : (tensor<100x97x1x62x44x97xf32>) -> tensor<100x97x1x62x44x97xf32>
    %10 = tosa.tanh %0 : (tensor<100x97x1x62x44x97xf32>) -> tensor<100x97x1x62x44x97xf32>
    %11 = tosa.logical_xor %6, %6 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %r_12 = tosa.const_shape {values = dense<[ 4, 11, 1, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %12 = tosa.reshape %2, %r_12 : (tensor<88xi1>, !tosa.shape<4>) -> tensor<4x11x1x2xi1>
    %13 = tosa.argmax %11 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %14 = tosa.reciprocal %0 : (tensor<100x97x1x62x44x97xf32>) -> tensor<100x97x1x62x44x97xf32>
    %15 = tosa.concat %12, %12 {axis = 3 : i32} : (tensor<4x11x1x2xi1>, tensor<4x11x1x2xi1>) -> tensor<4x11x1x4xi1>
    %16 = tosa.logical_right_shift %12, %12 : (tensor<4x11x1x2xi1>, tensor<4x11x1x2xi1>) -> tensor<4x11x1x2xi1>
    return %7, %8, %9, %10, %13, %14, %15, %16 : tensor<100x97x1x62x44x97xi1>, tensor<100x97x1x62x44x97xi1>, tensor<100x97x1x62x44x97xf32>, tensor<100x97x1x62x44x97xf32>, tensor<i32>, tensor<100x97x1x62x44x97xf32>, tensor<4x11x1x4xi1>, tensor<4x11x1x2xi1>
  }
}
