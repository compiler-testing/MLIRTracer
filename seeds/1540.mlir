module {
  func.func @main(%arg0: tensor<60x50xi64>, %arg1: tensor<50x83xi32>, %arg2: tensor<50x83xi32>, %arg3: tensor<87x15x43xf32>) -> (tensor<50x83xi32>, tensor<2x100xi64>, tensor<87x15x43xf32>) {
    %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<60x50xi64>) -> tensor<1x50xi64>
    %1 = tosa.add %0, %0 : (tensor<1x50xi64>, tensor<1x50xi64>) -> tensor<1x50xi64>
    %2 = tosa.concat %1, %1 {axis = 0 : i32} : (tensor<1x50xi64>, tensor<1x50xi64>) -> tensor<2x50xi64>
    %3 = tosa.intdiv %arg1, %arg2 : (tensor<50x83xi32>, tensor<50x83xi32>) -> tensor<50x83xi32>
    %4 = tosa.concat %2, %2 {axis = 1 : i32} : (tensor<2x50xi64>, tensor<2x50xi64>) -> tensor<2x100xi64>
    %5 = tosa.logical_right_shift %4, %4 : (tensor<2x100xi64>, tensor<2x100xi64>) -> tensor<2x100xi64>
    %6 = tosa.exp %arg3 : (tensor<87x15x43xf32>) -> tensor<87x15x43xf32>
    return %3, %5, %6 : tensor<50x83xi32>, tensor<2x100xi64>, tensor<87x15x43xf32>
  }
}
