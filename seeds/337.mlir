module {
  func.func @main(%arg0: tensor<16x11x9x37xf32>, %arg1: tensor<72x36xi1>) -> (tensor<16x11x9x74xi1>, tensor<72xi32>) {
    %0 = tosa.log %arg0 : (tensor<16x11x9x37xf32>) -> tensor<16x11x9x37xf32>
    %1 = tosa.pow %0, %0 : (tensor<16x11x9x37xf32>, tensor<16x11x9x37xf32>) -> tensor<16x11x9x37xf32>
    %2 = tosa.rsqrt %1 : (tensor<16x11x9x37xf32>) -> tensor<16x11x9x37xf32>
    %3 = tosa.concat %2, %2 {axis = 3 : i32} : (tensor<16x11x9x37xf32>, tensor<16x11x9x37xf32>) -> tensor<16x11x9x74xf32>
    %4 = tosa.reduce_any %arg1 {axis = 1 : i32} : (tensor<72x36xi1>) -> tensor<72x1xi1>
    %5 = tosa.greater_equal %3, %3 : (tensor<16x11x9x74xf32>, tensor<16x11x9x74xf32>) -> tensor<16x11x9x74xi1>
    %6 = tosa.argmax %4 {axis = 1 : i32} : (tensor<72x1xi1>) -> tensor<72xi32>
    %7 = tosa.arithmetic_right_shift %6, %6 {round = false} : (tensor<72xi32>, tensor<72xi32>) -> tensor<72xi32>
    %8 = tosa.logical_not %5 : (tensor<16x11x9x74xi1>) -> tensor<16x11x9x74xi1>
    %9 = tosa.bitwise_and %8, %8 : (tensor<16x11x9x74xi1>, tensor<16x11x9x74xi1>) -> tensor<16x11x9x74xi1>
    %10 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %11 = tosa.transpose %7 {perms = array<i32: 0>} : (tensor<72xi32>) -> tensor<72xi32>
    return %9, %11 : tensor<16x11x9x74xi1>, tensor<72xi32>
  }
}
