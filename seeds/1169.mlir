module {
  func.func @main(%arg0: tensor<4x32x82xi1>, %arg1: tensor<28xf32>, %arg2: tensor<11x42xi32>, %arg3: tensor<11x42xi32>) -> (tensor<1x32x82xi1>, tensor<28xf32>, tensor<11x42xi32>, tensor<231x1x2xi32>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<4x32x82xi1>) -> tensor<1x32x82xi1>
    %1 = tosa.bitwise_not %0 : (tensor<1x32x82xi1>) -> tensor<1x32x82xi1>
    %2 = tosa.sigmoid %arg1 : (tensor<28xf32>) -> tensor<28xf32>
    %3 = tosa.floor %2 : (tensor<28xf32>) -> tensor<28xf32>
    %4 = tosa.add %3, %2 : (tensor<28xf32>, tensor<28xf32>) -> tensor<28xf32>
    %5 = tosa.intdiv %arg2, %arg3 : (tensor<11x42xi32>, tensor<11x42xi32>) -> tensor<11x42xi32>
    %6 = tosa.logical_left_shift %5, %5 : (tensor<11x42xi32>, tensor<11x42xi32>) -> tensor<11x42xi32>
    %r_7 = tosa.const_shape {values = dense<[ 231, 1, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = tosa.reshape %5, %r_7 : (tensor<11x42xi32>, !tosa.shape<3>) -> tensor<231x1x2xi32>
    return %1, %4, %6, %7 : tensor<1x32x82xi1>, tensor<28xf32>, tensor<11x42xi32>, tensor<231x1x2xi32>
  }
}
