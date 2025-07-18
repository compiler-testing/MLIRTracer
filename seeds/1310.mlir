module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<i8>, %arg2: tensor<21x88x15xf32>) -> (tensor<21x88x1xf32>, tensor<i8>, tensor<1x88x15xf32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %1 = tosa.ceil %arg2 : (tensor<21x88x15xf32>) -> tensor<21x88x15xf32>
    %2 = tosa.reduce_sum %1 {axis = 2 : i32} : (tensor<21x88x15xf32>) -> tensor<21x88x1xf32>
    %3 = tosa.rsqrt %2 : (tensor<21x88x1xf32>) -> tensor<21x88x1xf32>
    %4 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<21x88x15xf32>) -> tensor<1x88x15xf32>
    %5 = tosa.logical_left_shift %0, %0 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %6 = tosa.reduce_sum %4 {axis = 0 : i32} : (tensor<1x88x15xf32>) -> tensor<1x88x15xf32>
    return %3, %5, %6 : tensor<21x88x1xf32>, tensor<i8>, tensor<1x88x15xf32>
  }
}
