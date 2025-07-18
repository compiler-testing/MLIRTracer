module {
  func.func @main(%arg0: tensor<3x24xi64>, %arg1: tensor<3x1xi64>, %arg2: tensor<82xi1>) -> (tensor<3x48xi64>, tensor<1xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<3x24xi64>, tensor<3x1xi64>) -> tensor<3x24xi64>
    %1 = tosa.reduce_all %arg2 {axis = 0 : i32} : (tensor<82xi1>) -> tensor<1xi1>
    %2 = tosa.concat %0, %0 {axis = 1 : i32} : (tensor<3x24xi64>, tensor<3x24xi64>) -> tensor<3x48xi64>
    %3 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %2, %3 : tensor<3x48xi64>, tensor<1xi1>
  }
}
