module {
  func.func @main(%arg0: tensor<30x44x45x46xi8>, %arg1: tensor<1x1x1x46xi8>) -> tensor<1x46x44xi1> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<30x44x45x46xi8>, tensor<1x1x1x46xi8>) -> tensor<30x44x45x46xi8>
    %1 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0, 3, 1, 2>} : (tensor<30x44x45x46xi8>) -> tensor<30x46x44x45xi8>
    %3 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<30x46x44x45xi8>) -> tensor<1x46x44x45xi8>
    %4 = tosa.argmax %3 {axis = 3 : i32} : (tensor<1x46x44x45xi8>) -> tensor<1x46x44xi32>
    %5 = tosa.greater %4, %4 : (tensor<1x46x44xi32>, tensor<1x46x44xi32>) -> tensor<1x46x44xi1>
    %6 = tosa.clamp %5 {min_val = 1 : i1, max_val = 1 : i1} : (tensor<1x46x44xi1>) -> tensor<1x46x44xi1>
    %7 = tosa.bitwise_and %6, %6 : (tensor<1x46x44xi1>, tensor<1x46x44xi1>) -> tensor<1x46x44xi1>
    %8 = tosa.clz %7 : (tensor<1x46x44xi1>) -> tensor<1x46x44xi1>
    %9 = tosa.logical_left_shift %8, %8 : (tensor<1x46x44xi1>, tensor<1x46x44xi1>) -> tensor<1x46x44xi1>
    return %9 : tensor<1x46x44xi1>
  }
}
