module {
  func.func @main(%arg0: tensor<44x71x17xi32>, %arg1: tensor<62x33x19x12xf32>, %arg2: tensor<30x32x8xi1>) -> (tensor<44x71x17xi32>, tensor<62x33x19x12xf32>, tensor<62x33x19x1xf32>, tensor<1x1x8xi1>) {
    %0 = tosa.clz %arg0 : (tensor<44x71x17xi32>) -> tensor<44x71x17xi32>
    %1 = tosa.sigmoid %arg1 : (tensor<62x33x19x12xf32>) -> tensor<62x33x19x12xf32>
    %2 = tosa.reduce_all %arg2 {axis = 1 : i32} : (tensor<30x32x8xi1>) -> tensor<30x1x8xi1>
    %3 = tosa.clz %0 : (tensor<44x71x17xi32>) -> tensor<44x71x17xi32>
    %4 = tosa.tanh %1 : (tensor<62x33x19x12xf32>) -> tensor<62x33x19x12xf32>
    %5 = tosa.reduce_max %1 {axis = 3 : i32} : (tensor<62x33x19x12xf32>) -> tensor<62x33x19x1xf32>
    %6 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<30x1x8xi1>) -> tensor<1x1x8xi1>
    return %3, %4, %5, %6 : tensor<44x71x17xi32>, tensor<62x33x19x12xf32>, tensor<62x33x19x1xf32>, tensor<1x1x8xi1>
  }
}
