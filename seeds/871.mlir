module {
  func.func @main(%arg0: tensor<17xf32>, %arg1: tensor<5x3x68x6xi16>, %arg2: tensor<5x1x1x1xi16>, %arg3: tensor<30x63x19xi1>) -> (tensor<5x3x68x6xi16>, tensor<i32>, tensor<1x1x19xi1>, tensor<30x1x38xi1>, tensor<17xf32>) {
    %0 = tosa.rsqrt %arg0 : (tensor<17xf32>) -> tensor<17xf32>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<17xf32>) -> tensor<1xf32>
    %2 = tosa.bitwise_xor %arg1, %arg2 : (tensor<5x3x68x6xi16>, tensor<5x1x1x1xi16>) -> tensor<5x3x68x6xi16>
    %3 = tosa.ceil %1 : (tensor<1xf32>) -> tensor<1xf32>
    %4 = tosa.argmax %3 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<i32>
    %5 = tosa.reduce_any %arg3 {axis = 1 : i32} : (tensor<30x63x19xi1>) -> tensor<30x1x19xi1>
    %6 = tosa.reduce_all %5 {axis = 0 : i32} : (tensor<30x1x19xi1>) -> tensor<1x1x19xi1>
    %7 = tosa.arithmetic_right_shift %6, %6 {round = false} : (tensor<1x1x19xi1>, tensor<1x1x19xi1>) -> tensor<1x1x19xi1>
    %8 = tosa.logical_not %5 : (tensor<30x1x19xi1>) -> tensor<30x1x19xi1>
    %9 = tosa.abs %7 : (tensor<1x1x19xi1>) -> tensor<1x1x19xi1>
    %10 = tosa.bitwise_and %9, %6 : (tensor<1x1x19xi1>, tensor<1x1x19xi1>) -> tensor<1x1x19xi1>
    %11 = tosa.concat %8, %5 {axis = 2 : i32} : (tensor<30x1x19xi1>, tensor<30x1x19xi1>) -> tensor<30x1x38xi1>
    %12 = tosa.sigmoid %0 : (tensor<17xf32>) -> tensor<17xf32>
    return %2, %4, %10, %11, %12 : tensor<5x3x68x6xi16>, tensor<i32>, tensor<1x1x19xi1>, tensor<30x1x38xi1>, tensor<17xf32>
  }
}
