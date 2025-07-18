module {
  func.func @main(%arg0: tensor<16x76xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<26x53x52xi32>) -> (tensor<16x1xf32>, tensor<16x76xf32>, tensor<26x53x52xi32>, tensor<26x1x52xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<16x76xf32>, tensor<1x1xf32>) -> tensor<16x76xf32>
    %1 = tosa.reduce_min %0 {axis = 1 : i32} : (tensor<16x76xf32>) -> tensor<16x1xf32>
    %2 = tosa.clz %arg2 : (tensor<26x53x52xi32>) -> tensor<26x53x52xi32>
    %3 = tosa.bitwise_xor %2, %2 : (tensor<26x53x52xi32>, tensor<26x53x52xi32>) -> tensor<26x53x52xi32>
    %4 = tosa.reduce_sum %2 {axis = 1 : i32} : (tensor<26x53x52xi32>) -> tensor<26x1x52xi32>
    %5 = tosa.bitwise_or %4, %4 : (tensor<26x1x52xi32>, tensor<26x1x52xi32>) -> tensor<26x1x52xi32>
    %6 = tosa.pow %0, %0 : (tensor<16x76xf32>, tensor<16x76xf32>) -> tensor<16x76xf32>
    %7 = tosa.add %3, %2 : (tensor<26x53x52xi32>, tensor<26x53x52xi32>) -> tensor<26x53x52xi32>
    %8 = tosa.greater_equal %5, %5 : (tensor<26x1x52xi32>, tensor<26x1x52xi32>) -> tensor<26x1x52xi1>
    return %1, %6, %7, %8 : tensor<16x1xf32>, tensor<16x76xf32>, tensor<26x53x52xi32>, tensor<26x1x52xi1>
  }
}
