module {
  func.func @main(%arg0: tensor<5x95x88x43xi32>, %arg1: tensor<5x1x1x43xi32>, %arg2: tensor<67x90x86x27x57xf32>) -> (tensor<1x95x88x43xi1>, tensor<67x90x86x27x57xi1>, tensor<57x27x90x86x67xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<5x95x88x43xi32>, tensor<5x1x1x43xi32>) -> tensor<5x95x88x43xi1>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<5x95x88x43xi1>) -> tensor<1x95x88x43xi1>
    %2 = tosa.reciprocal %arg2 : (tensor<67x90x86x27x57xf32>) -> tensor<67x90x86x27x57xf32>
    %3 = tosa.greater %2, %2 : (tensor<67x90x86x27x57xf32>, tensor<67x90x86x27x57xf32>) -> tensor<67x90x86x27x57xi1>
    %4 = tosa.clz %3 : (tensor<67x90x86x27x57xi1>) -> tensor<67x90x86x27x57xi1>
    %5 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %6 = tosa.transpose %2 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<67x90x86x27x57xf32>) -> tensor<57x27x90x86x67xf32>
    %7 = tosa.greater_equal %6, %6 : (tensor<57x27x90x86x67xf32>, tensor<57x27x90x86x67xf32>) -> tensor<57x27x90x86x67xi1>
    return %1, %4, %7 : tensor<1x95x88x43xi1>, tensor<67x90x86x27x57xi1>, tensor<57x27x90x86x67xi1>
  }
}
