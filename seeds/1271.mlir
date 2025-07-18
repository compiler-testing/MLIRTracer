module {
  func.func @main(%arg0: tensor<73x3xf32>) -> tensor<73x3xf32> {
    %0 = tosa.log %arg0 : (tensor<73x3xf32>) -> tensor<73x3xf32>
    %1 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0, 1>} : (tensor<73x3xf32>) -> tensor<73x3xf32>
    return %2 : tensor<73x3xf32>
  }
}
