module {
  func.func @main(%arg0: tensor<74x37xi16>, %arg1: tensor<49x96xi1>, %arg2: tensor<45x99x43xf32>) -> (tensor<37xi32>, tensor<45x99x43xf32>, tensor<1x2xi1>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<74x37xi16>) -> tensor<74x37xi16>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<74x37xi16>) -> tensor<1x37xi16>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<1x37xi16>) -> tensor<1x37xi16>
    %3 = tosa.argmax %2 {axis = 0 : i32} : (tensor<1x37xi16>) -> tensor<37xi32>
    %4 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<49x96xi1>) -> tensor<1x96xi1>
    %5 = tosa.rsqrt %arg2 : (tensor<45x99x43xf32>) -> tensor<45x99x43xf32>
    %6 = tosa.logical_right_shift %4, %4 : (tensor<1x96xi1>, tensor<1x96xi1>) -> tensor<1x96xi1>
    %s_7_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_7_size = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %7 = tosa.slice %6, %s_7_start, %s_7_size : (tensor<1x96xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x2xi1>
    return %3, %5, %7 : tensor<37xi32>, tensor<45x99x43xf32>, tensor<1x2xi1>
  }
}
