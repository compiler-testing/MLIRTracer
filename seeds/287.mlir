module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<26x27x14xf32>, %arg3: tensor<26x52xi1>, %arg4: tensor<26x1xi1>) -> (tensor<i16>, tensor<26x1x14xf32>, tensor<26x52xi1>, tensor<52xi32>, tensor<26x1x14xf32>, tensor<26x27x14xi1>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %1 = tosa.tanh %arg2 : (tensor<26x27x14xf32>) -> tensor<26x27x14xf32>
    %2 = tosa.abs %1 : (tensor<26x27x14xf32>) -> tensor<26x27x14xf32>
    %3 = tosa.rsqrt %2 : (tensor<26x27x14xf32>) -> tensor<26x27x14xf32>
    %4 = tosa.reduce_max %3 {axis = 1 : i32} : (tensor<26x27x14xf32>) -> tensor<26x1x14xf32>
    %5 = tosa.minimum %4, %4 : (tensor<26x1x14xf32>, tensor<26x1x14xf32>) -> tensor<26x1x14xf32>
    %6 = tosa.logical_xor %arg3, %arg4 : (tensor<26x52xi1>, tensor<26x1xi1>) -> tensor<26x52xi1>
    %7 = tosa.logical_xor %6, %6 : (tensor<26x52xi1>, tensor<26x52xi1>) -> tensor<26x52xi1>
    %8 = tosa.greater %2, %2 : (tensor<26x27x14xf32>, tensor<26x27x14xf32>) -> tensor<26x27x14xi1>
    %9 = tosa.argmax %6 {axis = 0 : i32} : (tensor<26x52xi1>) -> tensor<52xi32>
    %10 = tosa.pow %4, %4 : (tensor<26x1x14xf32>, tensor<26x1x14xf32>) -> tensor<26x1x14xf32>
    %11 = tosa.clz %8 : (tensor<26x27x14xi1>) -> tensor<26x27x14xi1>
    %12 = tosa.reverse %11 {axis = 1 : i32} : (tensor<26x27x14xi1>) -> tensor<26x27x14xi1>
    return %0, %5, %7, %9, %10, %12 : tensor<i16>, tensor<26x1x14xf32>, tensor<26x52xi1>, tensor<52xi32>, tensor<26x1x14xf32>, tensor<26x27x14xi1>
  }
}
