module {
  func.func @main(%arg0: tensor<19x47x19xf32>, %arg1: tensor<1x47x19xf32>, %arg2: tensor<52x67x82x95xi1>) -> (tensor<19x1x19xf32>, tensor<1x1xi32>, tensor<19x47x19xf32>, tensor<52x67x82x1xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<19x47x19xf32>, tensor<1x47x19xf32>) -> tensor<19x47x19xf32>
    %1 = tosa.minimum %0, %0 : (tensor<19x47x19xf32>, tensor<19x47x19xf32>) -> tensor<19x47x19xf32>
    %2 = tosa.floor %1 : (tensor<19x47x19xf32>) -> tensor<19x47x19xf32>
    %3 = tosa.reduce_max %2 {axis = 1 : i32} : (tensor<19x47x19xf32>) -> tensor<19x1x19xf32>
    %4 = tosa.argmax %3 {axis = 0 : i32} : (tensor<19x1x19xf32>) -> tensor<1x19xi32>
    %5 = tosa.logical_right_shift %4, %4 : (tensor<1x19xi32>, tensor<1x19xi32>) -> tensor<1x19xi32>
    %6 = tosa.pow %3, %3 : (tensor<19x1x19xf32>, tensor<19x1x19xf32>) -> tensor<19x1x19xf32>
    %7 = tosa.reduce_all %arg2 {axis = 3 : i32} : (tensor<52x67x82x95xi1>) -> tensor<52x67x82x1xi1>
    %8 = tosa.reduce_max %5 {axis = 1 : i32} : (tensor<1x19xi32>) -> tensor<1x1xi32>
    %9 = tosa.logical_not %7 : (tensor<52x67x82x1xi1>) -> tensor<52x67x82x1xi1>
    %10 = tosa.exp %1 : (tensor<19x47x19xf32>) -> tensor<19x47x19xf32>
    %11 = tosa.add %9, %9 : (tensor<52x67x82x1xi1>, tensor<52x67x82x1xi1>) -> tensor<52x67x82x1xi1>
    return %6, %8, %10, %11 : tensor<19x1x19xf32>, tensor<1x1xi32>, tensor<19x47x19xf32>, tensor<52x67x82x1xi1>
  }
}
