module {
  func.func @main(%arg0: tensor<63xf32>, %arg1: tensor<68x58x84x87xi1>, %arg2: tensor<1x58x1x1xi1>) -> (tensor<1xi1>, tensor<1xf32>, tensor<1x58x84x1xi1>, tensor<68x58x84x87xi1>) {
    %0 = tosa.abs %arg0 : (tensor<63xf32>) -> tensor<63xf32>
    %1 = tosa.sub %0, %0 : (tensor<63xf32>, tensor<63xf32>) -> tensor<63xf32>
    %2 = tosa.rsqrt %1 : (tensor<63xf32>) -> tensor<63xf32>
    %3 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %4 = tosa.transpose %2 {perms = array<i32: 0>} : (tensor<63xf32>) -> tensor<63xf32>
    %5 = tosa.reduce_max %4 {axis = 0 : i32} : (tensor<63xf32>) -> tensor<1xf32>
    %6 = tosa.sub %5, %5 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %7 = tosa.ceil %6 : (tensor<1xf32>) -> tensor<1xf32>
    %8 = tosa.logical_xor %arg1, %arg2 : (tensor<68x58x84x87xi1>, tensor<1x58x1x1xi1>) -> tensor<68x58x84x87xi1>
    %9 = tosa.add %8, %8 : (tensor<68x58x84x87xi1>, tensor<68x58x84x87xi1>) -> tensor<68x58x84x87xi1>
    %10 = tosa.logical_left_shift %9, %8 : (tensor<68x58x84x87xi1>, tensor<68x58x84x87xi1>) -> tensor<68x58x84x87xi1>
    %11 = tosa.logical_or %10, %8 : (tensor<68x58x84x87xi1>, tensor<68x58x84x87xi1>) -> tensor<68x58x84x87xi1>
    %12 = tosa.clz %11 : (tensor<68x58x84x87xi1>) -> tensor<68x58x84x87xi1>
    %13 = tosa.equal %7, %6 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>
    %14 = tosa.reduce_all %12 {axis = 3 : i32} : (tensor<68x58x84x87xi1>) -> tensor<68x58x84x1xi1>
    %15 = tosa.reciprocal %6 : (tensor<1xf32>) -> tensor<1xf32>
    %16 = tosa.reduce_any %14 {axis = 0 : i32} : (tensor<68x58x84x1xi1>) -> tensor<1x58x84x1xi1>
    %17 = tosa.sub %9, %9 : (tensor<68x58x84x87xi1>, tensor<68x58x84x87xi1>) -> tensor<68x58x84x87xi1>
    return %13, %15, %16, %17 : tensor<1xi1>, tensor<1xf32>, tensor<1x58x84x1xi1>, tensor<68x58x84x87xi1>
  }
}
