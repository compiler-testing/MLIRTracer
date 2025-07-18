module {
  func.func @main(%arg0: tensor<49xi64>, %arg1: tensor<94xf32>) -> (tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i32>, tensor<1xf32>, tensor<94xf32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<49xi64>) -> tensor<i32>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.greater %1, %0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = tosa.clz %2 : (tensor<i1>) -> tensor<i1>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %5 = tosa.rsqrt %arg1 : (tensor<94xf32>) -> tensor<94xf32>
    %6 = tosa.clz %4 : (tensor<i1>) -> tensor<i1>
    %7 = tosa.clz %6 : (tensor<i1>) -> tensor<i1>
    %8 = tosa.argmax %5 {axis = 0 : i32} : (tensor<94xf32>) -> tensor<i32>
    %9 = tosa.reduce_max %5 {axis = 0 : i32} : (tensor<94xf32>) -> tensor<1xf32>
    %10 = tosa.bitwise_or %3, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %11 = tosa.tanh %9 : (tensor<1xf32>) -> tensor<1xf32>
    %12 = tosa.bitwise_not %3 : (tensor<i1>) -> tensor<i1>
    %13 = tosa.sigmoid %5 : (tensor<94xf32>) -> tensor<94xf32>
    %14 = tosa.clz %2 : (tensor<i1>) -> tensor<i1>
    %15 = tosa.reciprocal %13 : (tensor<94xf32>) -> tensor<94xf32>
    %16 = tosa.intdiv %8, %8 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %17 = tosa.reduce_product %11 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %18 = tosa.abs %15 : (tensor<94xf32>) -> tensor<94xf32>
    return %7, %10, %12, %14, %16, %17, %18 : tensor<i1>, tensor<i1>, tensor<i1>, tensor<i1>, tensor<i32>, tensor<1xf32>, tensor<94xf32>
  }
}
