module {
  func.func @main(%arg0: tensor<37x44xf32>) -> tensor<37x1xf32> {
    %0 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 1>} : (tensor<37x44xf32>) -> tensor<37x44xf32>
    %2 = tosa.floor %1 : (tensor<37x44xf32>) -> tensor<37x44xf32>
    %3 = tosa.reduce_max %2 {axis = 1 : i32} : (tensor<37x44xf32>) -> tensor<37x1xf32>
    return %3 : tensor<37x1xf32>
  }
}
