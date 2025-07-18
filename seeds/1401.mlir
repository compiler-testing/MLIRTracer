module {
  func.func @main(%arg0: tensor<53xi64>, %arg1: tensor<89x63xf32>, %arg2: tensor<96x21x34x50xi1>) -> (tensor<96x21x34x50xi1>, tensor<63xi32>, tensor<1xi64>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<53xi64>) -> tensor<1xi64>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<1xi64>) -> tensor<1xi64>
    %2 = tosa.log %arg1 : (tensor<89x63xf32>) -> tensor<89x63xf32>
    %3 = tosa.logical_not %arg2 : (tensor<96x21x34x50xi1>) -> tensor<96x21x34x50xi1>
    %4 = tosa.argmax %2 {axis = 0 : i32} : (tensor<89x63xf32>) -> tensor<63xi32>
    %5 = tosa.clz %1 : (tensor<1xi64>) -> tensor<1xi64>
    return %3, %4, %5 : tensor<96x21x34x50xi1>, tensor<63xi32>, tensor<1xi64>
  }
}
