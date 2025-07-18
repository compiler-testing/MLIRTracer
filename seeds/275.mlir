module {
  func.func @main(%arg0: tensor<97xi32>, %arg1: tensor<30x51xi1>) -> (tensor<1xi32>, tensor<1x51xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<97xi32>) -> tensor<1xi32>
    %1 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<30x51xi1>) -> tensor<1x51xi1>
    %2 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 0, 1>} : (tensor<1x51xi1>) -> tensor<1x51xi1>
    return %0, %3 : tensor<1xi32>, tensor<1x51xi1>
  }
}
