module {
  func.func @main(%arg0: tensor<42x72x10xi32>, %arg1: tensor<1x72x1xi32>) -> tensor<42x10x72xi32> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<42x72x10xi32>, tensor<1x72x1xi32>) -> tensor<42x72x10xi32>
    %1 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0, 2, 1>} : (tensor<42x72x10xi32>) -> tensor<42x10x72xi32>
    return %2 : tensor<42x10x72xi32>
  }
}
