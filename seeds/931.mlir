module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<31x19xf32>, %arg2: tensor<95xi1>, %arg3: tensor<67x26x84xi32>, %arg4: tensor<1x1x84xi32>) -> (tensor<f32>, tensor<95xi1>, tensor<1x19xi1>, tensor<67x26x84xi1>, tensor<67x26x84xi1>, tensor<67x26x1xi32>, tensor<67x26x84xi32>) {
    %0 = tosa.reciprocal %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.sub %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = tosa.reduce_max %arg1 {axis = 0 : i32} : (tensor<31x19xf32>) -> tensor<1x19xf32>
    %3 = tosa.clz %arg2 : (tensor<95xi1>) -> tensor<95xi1>
    %4 = tosa.intdiv %arg3, %arg4 : (tensor<67x26x84xi32>, tensor<1x1x84xi32>) -> tensor<67x26x84xi32>
    %5 = tosa.add %4, %4 : (tensor<67x26x84xi32>, tensor<67x26x84xi32>) -> tensor<67x26x84xi32>
    %6 = tosa.greater %2, %2 : (tensor<1x19xf32>, tensor<1x19xf32>) -> tensor<1x19xi1>
    %7 = tosa.bitwise_or %5, %5 : (tensor<67x26x84xi32>, tensor<67x26x84xi32>) -> tensor<67x26x84xi32>
    %8 = tosa.bitwise_or %6, %6 : (tensor<1x19xi1>, tensor<1x19xi1>) -> tensor<1x19xi1>
    %9 = tosa.bitwise_and %8, %8 : (tensor<1x19xi1>, tensor<1x19xi1>) -> tensor<1x19xi1>
    %10 = tosa.greater_equal %7, %5 : (tensor<67x26x84xi32>, tensor<67x26x84xi32>) -> tensor<67x26x84xi1>
    %11 = tosa.clz %7 : (tensor<67x26x84xi32>) -> tensor<67x26x84xi32>
    %12 = tosa.equal %11, %11 : (tensor<67x26x84xi32>, tensor<67x26x84xi32>) -> tensor<67x26x84xi1>
    %13 = tosa.reduce_min %5 {axis = 2 : i32} : (tensor<67x26x84xi32>) -> tensor<67x26x1xi32>
    %14 = tosa.clamp %7 {min_val = -45 : i32, max_val = -36 : i32} : (tensor<67x26x84xi32>) -> tensor<67x26x84xi32>
    return %1, %3, %9, %10, %12, %13, %14 : tensor<f32>, tensor<95xi1>, tensor<1x19xi1>, tensor<67x26x84xi1>, tensor<67x26x84xi1>, tensor<67x26x1xi32>, tensor<67x26x84xi32>
  }
}
