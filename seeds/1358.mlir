module {
  func.func @main(%arg0: tensor<96x62x71x81x93xi64>, %arg1: tensor<1x100x53x44xi1>) -> (tensor<93x81x62x71x96xi1>, tensor<1x100x53x44xi1>) {
    %0 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<96x62x71x81x93xi64>) -> tensor<93x81x62x71x96xi64>
    %2 = tosa.arithmetic_right_shift %1, %1 {round = false} : (tensor<93x81x62x71x96xi64>, tensor<93x81x62x71x96xi64>) -> tensor<93x81x62x71x96xi64>
    %3 = tosa.equal %2, %1 : (tensor<93x81x62x71x96xi64>, tensor<93x81x62x71x96xi64>) -> tensor<93x81x62x71x96xi1>
    %4 = tosa.logical_xor %3, %3 : (tensor<93x81x62x71x96xi1>, tensor<93x81x62x71x96xi1>) -> tensor<93x81x62x71x96xi1>
    %5 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<1x100x53x44xi1>) -> tensor<1x100x53x44xi1>
    return %4, %5 : tensor<93x81x62x71x96xi1>, tensor<1x100x53x44xi1>
  }
}
