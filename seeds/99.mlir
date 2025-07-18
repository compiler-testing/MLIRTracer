module {
  func.func @main(%arg0: tensor<92x25xi8>, %arg1: tensor<92x25xi8>, %arg2: tensor<26x19x74x41xf32>) -> (tensor<1x25xi8>, tensor<1x1x74x41xi1>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<92x25xi8>, tensor<92x25xi8>) -> tensor<92x25xi8>
    %1 = tosa.sub %0, %0 : (tensor<92x25xi8>, tensor<92x25xi8>) -> tensor<92x25xi8>
    %2 = tosa.arithmetic_right_shift %1, %0 {round = false} : (tensor<92x25xi8>, tensor<92x25xi8>) -> tensor<92x25xi8>
    %3 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<92x25xi8>) -> tensor<1x25xi8>
    %4 = tosa.rsqrt %arg2 : (tensor<26x19x74x41xf32>) -> tensor<26x19x74x41xf32>
    %5 = tosa.greater_equal %4, %4 : (tensor<26x19x74x41xf32>, tensor<26x19x74x41xf32>) -> tensor<26x19x74x41xi1>
    %6 = tosa.reduce_any %5 {axis = 1 : i32} : (tensor<26x19x74x41xi1>) -> tensor<26x1x74x41xi1>
    %7 = tosa.reduce_sum %6 {axis = 0 : i32} : (tensor<26x1x74x41xi1>) -> tensor<1x1x74x41xi1>
    return %3, %7 : tensor<1x25xi8>, tensor<1x1x74x41xi1>
  }
}
