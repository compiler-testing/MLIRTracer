module {
  func.func @main(%arg0: tensor<5x2x97xf32>, %arg1: tensor<5x1x1xf32>, %arg2: tensor<57xf32>, %arg3: tensor<57xf32>) -> (tensor<1xf32>, tensor<57xf32>, tensor<1x1x97xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<5x2x97xf32>, tensor<5x1x1xf32>) -> tensor<5x2x97xi1>
    %1 = tosa.pow %arg2, %arg3 : (tensor<57xf32>, tensor<57xf32>) -> tensor<57xf32>
    %2 = tosa.reduce_any %0 {axis = 1 : i32} : (tensor<5x2x97xi1>) -> tensor<5x1x97xi1>
    %3 = tosa.floor %1 : (tensor<57xf32>) -> tensor<57xf32>
    %4 = tosa.reduce_min %3 {axis = 0 : i32} : (tensor<57xf32>) -> tensor<1xf32>
    %5 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %6 = tosa.transpose %4 {perms = array<i32: 0>} : (tensor<1xf32>) -> tensor<1xf32>
    %7 = tosa.tanh %6 : (tensor<1xf32>) -> tensor<1xf32>
    %8 = tosa.maximum %1, %3 : (tensor<57xf32>, tensor<57xf32>) -> tensor<57xf32>
    %9 = tosa.reduce_all %2 {axis = 0 : i32} : (tensor<5x1x97xi1>) -> tensor<1x1x97xi1>
    return %7, %8, %9 : tensor<1xf32>, tensor<57xf32>, tensor<1x1x97xi1>
  }
}
