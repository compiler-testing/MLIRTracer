module {
  func.func @main(%arg0: tensor<39x77x26xi1>) -> tensor<39x26x77xi1> {
    %0 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 1>} : (tensor<39x77x26xi1>) -> tensor<39x26x77xi1>
    return %1 : tensor<39x26x77xi1>
  }
}
