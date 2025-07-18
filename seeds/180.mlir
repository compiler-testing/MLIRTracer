module {
  func.func @main(%arg0: tensor<22x2xf32>, %arg1: tensor<46x5xi1>) -> (tensor<22x2xf32>, tensor<46x1xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<22x2xf32>) -> tensor<22x2xf32>
    %1 = tosa.reciprocal %0 : (tensor<22x2xf32>) -> tensor<22x2xf32>
    %2 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 0, 1>} : (tensor<22x2xf32>) -> tensor<22x2xf32>
    %4 = tosa.reduce_all %arg1 {axis = 1 : i32} : (tensor<46x5xi1>) -> tensor<46x1xi1>
    return %3, %4 : tensor<22x2xf32>, tensor<46x1xi1>
  }
}
