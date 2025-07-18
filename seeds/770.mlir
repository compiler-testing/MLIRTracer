module {
  func.func @main(%arg0: tensor<79xf32>, %arg1: tensor<32x85x12xi1>) -> (tensor<32x1x12xi1>, tensor<79xf32>) {
    %0 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 0>} : (tensor<79xf32>) -> tensor<79xf32>
    %2 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<32x85x12xi1>) -> tensor<32x1x12xi1>
    %3 = tosa.log %1 : (tensor<79xf32>) -> tensor<79xf32>
    %4 = tosa.sub %3, %3 : (tensor<79xf32>, tensor<79xf32>) -> tensor<79xf32>
    %5 = tosa.rsqrt %4 : (tensor<79xf32>) -> tensor<79xf32>
    %6 = tosa.tanh %5 : (tensor<79xf32>) -> tensor<79xf32>
    %7 = tosa.minimum %6, %5 : (tensor<79xf32>, tensor<79xf32>) -> tensor<79xf32>
    %8 = tosa.floor %7 : (tensor<79xf32>) -> tensor<79xf32>
    %9 = tosa.reciprocal %8 : (tensor<79xf32>) -> tensor<79xf32>
    %10 = tosa.ceil %9 : (tensor<79xf32>) -> tensor<79xf32>
    return %2, %10 : tensor<32x1x12xi1>, tensor<79xf32>
  }
}
