module {
  func.func @main(%arg0: tensor<45x62x73x79xf32>) -> tensor<1x1x158xi32> {
    %t_0 = tosa.const_shape {values = dense<[ 2, 1, 3, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<45x62x73x79xf32>, !tosa.shape<4>) -> tensor<90x62x219x158xf32>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<90x62x219x158xf32>) -> tensor<90x1x219x158xf32>
    %2 = tosa.sub %1, %1 : (tensor<90x1x219x158xf32>, tensor<90x1x219x158xf32>) -> tensor<90x1x219x158xf32>
    %3 = tosa.argmax %2 {axis = 2 : i32} : (tensor<90x1x219x158xf32>) -> tensor<90x1x158xi32>
    %4 = tosa.arithmetic_right_shift %3, %3 {round = true} : (tensor<90x1x158xi32>, tensor<90x1x158xi32>) -> tensor<90x1x158xi32>
    %5 = tosa.bitwise_not %4 : (tensor<90x1x158xi32>) -> tensor<90x1x158xi32>
    %6 = tosa.reduce_sum %5 {axis = 0 : i32} : (tensor<90x1x158xi32>) -> tensor<1x1x158xi32>
    return %6 : tensor<1x1x158xi32>
  }
}
