module {
  func.func @main(%arg0: tensor<34xf32>, %arg1: tensor<1xf32>) -> (tensor<i1>, tensor<34xi1>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<34xf32>, tensor<1xf32>) -> tensor<34xf32>
    %1 = tosa.minimum %0, %0 : (tensor<34xf32>, tensor<34xf32>) -> tensor<34xf32>
    %2 = tosa.greater_equal %1, %0 : (tensor<34xf32>, tensor<34xf32>) -> tensor<34xi1>
    %3 = tosa.argmax %2 {axis = 0 : i32} : (tensor<34xi1>) -> tensor<i32>
    %4 = tosa.greater %3, %3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = tosa.logical_and %4, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %6 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %7 = tosa.transpose %2 {perms = array<i32: 0>} : (tensor<34xi1>) -> tensor<34xi1>
    return %5, %7 : tensor<i1>, tensor<34xi1>
  }
}
