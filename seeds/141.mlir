module {
  func.func @main(%arg0: tensor<56x21xi1>) -> tensor<21x1xi1> {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<56x21xi1>) -> tensor<1x21xi1>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<1x21xi1>, tensor<1x21xi1>) -> tensor<1x21xi1>
    %2 = "tosa.const"() {values = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 1, 0>} : (tensor<1x21xi1>) -> tensor<21x1xi1>
    return %3 : tensor<21x1xi1>
  }
}
