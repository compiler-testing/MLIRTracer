module {
  func.func @main(%arg0: tensor<17x36xf32>, %arg1: tensor<66x55xi1>, %arg2: tensor<66x55xi1>) -> (tensor<17x36xf32>, tensor<66x55xi1>, tensor<1x55xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<17x36xf32>) -> tensor<17x36xf32>
    %1 = tosa.sigmoid %0 : (tensor<17x36xf32>) -> tensor<17x36xf32>
    %2 = tosa.ceil %1 : (tensor<17x36xf32>) -> tensor<17x36xf32>
    %3 = tosa.logical_or %arg1, %arg2 : (tensor<66x55xi1>, tensor<66x55xi1>) -> tensor<66x55xi1>
    %4 = tosa.logical_or %3, %3 : (tensor<66x55xi1>, tensor<66x55xi1>) -> tensor<66x55xi1>
    %5 = tosa.identity %3 : (tensor<66x55xi1>) -> tensor<66x55xi1>
    %6 = tosa.logical_and %5, %5 : (tensor<66x55xi1>, tensor<66x55xi1>) -> tensor<66x55xi1>
    %7 = tosa.pow %2, %2 : (tensor<17x36xf32>, tensor<17x36xf32>) -> tensor<17x36xf32>
    %8 = tosa.add %6, %5 : (tensor<66x55xi1>, tensor<66x55xi1>) -> tensor<66x55xi1>
    %9 = tosa.log %7 : (tensor<17x36xf32>) -> tensor<17x36xf32>
    %10 = tosa.bitwise_and %4, %5 : (tensor<66x55xi1>, tensor<66x55xi1>) -> tensor<66x55xi1>
    %11 = tosa.reduce_max %8 {axis = 0 : i32} : (tensor<66x55xi1>) -> tensor<1x55xi1>
    return %9, %10, %11 : tensor<17x36xf32>, tensor<66x55xi1>, tensor<1x55xi1>
  }
}
