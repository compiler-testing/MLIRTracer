module {
  func.func @main(%arg0: tensor<100x40x16x21x46xi1>, %arg1: tensor<100x1x1x21x1xi1>, %arg2: tensor<57x39x92xf32>) -> (tensor<100x40x16x21x46xi1>, tensor<57x92x39xf32>, tensor<1x1x1xf32>, tensor<100x40x16x21x46xi1>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<100x40x16x21x46xi1>, tensor<100x1x1x21x1xi1>) -> tensor<100x40x16x21x46xi1>
    %1 = tosa.floor %arg2 : (tensor<57x39x92xf32>) -> tensor<57x39x92xf32>
    %2 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<57x39x92xf32>) -> tensor<57x1x92xf32>
    %3 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %4 = tosa.transpose %1 {perms = array<i32: 0, 2, 1>} : (tensor<57x39x92xf32>) -> tensor<57x92x39xf32>
    %5 = tosa.add %2, %2 : (tensor<57x1x92xf32>, tensor<57x1x92xf32>) -> tensor<57x1x92xf32>
    %6 = tosa.maximum %4, %4 : (tensor<57x92x39xf32>, tensor<57x92x39xf32>) -> tensor<57x92x39xf32>
    %7 = tosa.bitwise_not %0 : (tensor<100x40x16x21x46xi1>) -> tensor<100x40x16x21x46xi1>
    %8 = tosa.abs %6 : (tensor<57x92x39xf32>) -> tensor<57x92x39xf32>
    %9 = tosa.reduce_sum %5 {axis = 0 : i32} : (tensor<57x1x92xf32>) -> tensor<1x1x92xf32>
    %10 = tosa.reduce_product %9 {axis = 2 : i32} : (tensor<1x1x92xf32>) -> tensor<1x1x1xf32>
    %11 = tosa.logical_left_shift %0, %0 : (tensor<100x40x16x21x46xi1>, tensor<100x40x16x21x46xi1>) -> tensor<100x40x16x21x46xi1>
    return %7, %8, %10, %11 : tensor<100x40x16x21x46xi1>, tensor<57x92x39xf32>, tensor<1x1x1xf32>, tensor<100x40x16x21x46xi1>
  }
}
