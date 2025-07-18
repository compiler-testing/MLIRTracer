module {
  func.func @main(%arg0: tensor<94x75x22x37x61x13xi1>, %arg1: tensor<94x75x22x1x61x1xi1>, %arg2: tensor<42x14xi32>, %arg3: tensor<63xf32>) -> (tensor<94x75x22x37x61x13xi1>, tensor<42x1xi1>, tensor<63xf32>, tensor<1x1xi1>, tensor<63xf32>, tensor<1x1xi1>, tensor<1xi32>, tensor<1xi32>, tensor<63xf32>, tensor<63xf32>, tensor<1xi32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<94x75x22x37x61x13xi1>, tensor<94x75x22x1x61x1xi1>) -> tensor<94x75x22x37x61x13xi1>
    %1 = tosa.bitwise_or %0, %0 : (tensor<94x75x22x37x61x13xi1>, tensor<94x75x22x37x61x13xi1>) -> tensor<94x75x22x37x61x13xi1>
    %2 = tosa.reduce_max %arg2 {axis = 1 : i32} : (tensor<42x14xi32>) -> tensor<42x1xi32>
    %3 = tosa.ceil %arg3 : (tensor<63xf32>) -> tensor<63xf32>
    %4 = tosa.equal %2, %2 : (tensor<42x1xi32>, tensor<42x1xi32>) -> tensor<42x1xi1>
    %5 = tosa.minimum %3, %3 : (tensor<63xf32>, tensor<63xf32>) -> tensor<63xf32>
    %6 = tosa.bitwise_xor %4, %4 : (tensor<42x1xi1>, tensor<42x1xi1>) -> tensor<42x1xi1>
    %7 = tosa.reduce_all %4 {axis = 0 : i32} : (tensor<42x1xi1>) -> tensor<1x1xi1>
    %8 = tosa.rsqrt %3 : (tensor<63xf32>) -> tensor<63xf32>
    %9 = tosa.logical_right_shift %7, %7 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    %10 = tosa.reciprocal %3 : (tensor<63xf32>) -> tensor<63xf32>
    %11 = tosa.reduce_any %7 {axis = 1 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %12 = tosa.argmax %7 {axis = 1 : i32} : (tensor<1x1xi1>) -> tensor<1xi32>
    %13 = tosa.intdiv %12, %12 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %14 = tosa.reduce_sum %12 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %15 = tosa.sigmoid %3 : (tensor<63xf32>) -> tensor<63xf32>
    %16 = tosa.ceil %15 : (tensor<63xf32>) -> tensor<63xf32>
    %17 = tosa.tanh %5 : (tensor<63xf32>) -> tensor<63xf32>
    %18 = tosa.arithmetic_right_shift %12, %12 {round = false} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    return %1, %6, %8, %9, %10, %11, %13, %14, %16, %17, %18 : tensor<94x75x22x37x61x13xi1>, tensor<42x1xi1>, tensor<63xf32>, tensor<1x1xi1>, tensor<63xf32>, tensor<1x1xi1>, tensor<1xi32>, tensor<1xi32>, tensor<63xf32>, tensor<63xf32>, tensor<1xi32>
  }
}
