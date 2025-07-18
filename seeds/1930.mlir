module {
  func.func @main(%arg0: tensor<71x1x18xi32>, %arg1: tensor<71x18x95xi32>, %arg2: tensor<21x60x69x60x11x71xf32>) -> (tensor<71x1x95xi32>, tensor<21x60x69x60x11x71xf32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<71x1x18xi32>, tensor<71x18x95xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<71x1x95xi32>
    %1 = tosa.log %arg2 : (tensor<21x60x69x60x11x71xf32>) -> tensor<21x60x69x60x11x71xf32>
    %2 = tosa.intdiv %0, %0 : (tensor<71x1x95xi32>, tensor<71x1x95xi32>) -> tensor<71x1x95xi32>
    %3 = tosa.log %1 : (tensor<21x60x69x60x11x71xf32>) -> tensor<21x60x69x60x11x71xf32>
    return %2, %3 : tensor<71x1x95xi32>, tensor<21x60x69x60x11x71xf32>
  }
}
