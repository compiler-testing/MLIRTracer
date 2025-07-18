module {
  func.func @main(%arg0: tensor<70x89xi1>, %arg1: tensor<70x89xi1>) -> tensor<70x89xi1> {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<70x89xi1>, tensor<70x89xi1>) -> tensor<70x89xi1>
    %1 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0, 1>} : (tensor<70x89xi1>) -> tensor<70x89xi1>
    return %2 : tensor<70x89xi1>
  }
}
