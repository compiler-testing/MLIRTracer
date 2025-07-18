module {
  func.func @main(%arg0: tensor<65x41x11x19x79x50xi1>, %arg1: tensor<49x23x75x21xf32>) -> (tensor<49x23x75x21xf32>, tensor<79x19x41x11x65x50xi1>) {
    %0 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<65x41x11x19x79x50xi1>) -> tensor<79x19x41x11x65x50xi1>
    %2 = tosa.rsqrt %arg1 : (tensor<49x23x75x21xf32>) -> tensor<49x23x75x21xf32>
    %3 = tosa.clz %1 : (tensor<79x19x41x11x65x50xi1>) -> tensor<79x19x41x11x65x50xi1>
    return %2, %3 : tensor<49x23x75x21xf32>, tensor<79x19x41x11x65x50xi1>
  }
}
