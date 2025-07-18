module {
  func.func @main(%arg0: tensor<38xi1>, %arg1: tensor<26x80x45xf32>) -> (tensor<i32>, tensor<1xi1>, tensor<1xi1>, tensor<12x12xi32>, tensor<26x80x45xf32>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<38xi1>) -> tensor<1xi1>
    %1 = tosa.tanh %arg1 : (tensor<26x80x45xf32>) -> tensor<26x80x45xf32>
    %2 = tosa.logical_left_shift %0, %0 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %3 = tosa.argmax %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %4 = tosa.sub %1, %1 : (tensor<26x80x45xf32>, tensor<26x80x45xf32>) -> tensor<26x80x45xf32>
    %5 = tosa.logical_xor %2, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %s_6_start = tosa.const_shape {values = dense<[ 1, 19, 13 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_6_size = tosa.const_shape {values = dense<[ 12, 9, 12 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.slice %4, %s_6_start, %s_6_size : (tensor<26x80x45xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<12x9x12xf32>
    %7 = tosa.reverse %6 {axis = 2 : i32} : (tensor<12x9x12xf32>) -> tensor<12x9x12xf32>
    %8 = tosa.exp %7 : (tensor<12x9x12xf32>) -> tensor<12x9x12xf32>
    %9 = tosa.add %4, %4 : (tensor<26x80x45xf32>, tensor<26x80x45xf32>) -> tensor<26x80x45xf32>
    %10 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %11 = tosa.reduce_any %5 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.arithmetic_right_shift %11, %11 {round = false} : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %13 = tosa.logical_right_shift %10, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.argmax %8 {axis = 1 : i32} : (tensor<12x9x12xf32>) -> tensor<12x12xi32>
    %15 = tosa.tanh %9 : (tensor<26x80x45xf32>) -> tensor<26x80x45xf32>
    return %3, %12, %13, %14, %15 : tensor<i32>, tensor<1xi1>, tensor<1xi1>, tensor<12x12xi32>, tensor<26x80x45xf32>
  }
}
