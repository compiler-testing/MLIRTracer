module {
  func.func @main(%arg0: tensor<73x99xf32>) -> tensor<73x99xf32> {
    %0 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 1>} : (tensor<73x99xf32>) -> tensor<73x99xf32>
    return %1 : tensor<73x99xf32>
  }
}
