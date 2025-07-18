module {
  func.func @main(%arg0: tensor<48x80x49x79xi64>, %arg1: tensor<48x38x49x79xi64>, %arg2: tensor<33x63x44x82x48x12xf32>, %arg3: tensor<1x63x1x1x1x1xf32>, %arg4: tensor<80xi1>) -> (tensor<1x118x49x1xi64>, tensor<1x118x49x1xi64>, tensor<33x63x44x82x48x12xf32>, tensor<1xi1>, tensor<33x63x44x82x48x12xf32>, tensor<1xi1>, tensor<1xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<48x80x49x79xi64>, tensor<48x38x49x79xi64>) -> tensor<48x118x49x79xi64>
    %1 = tosa.pow %arg2, %arg3 : (tensor<33x63x44x82x48x12xf32>, tensor<1x63x1x1x1x1xf32>) -> tensor<33x63x44x82x48x12xf32>
    %2 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<48x118x49x79xi64>) -> tensor<1x118x49x79xi64>
    %3 = tosa.pow %1, %1 : (tensor<33x63x44x82x48x12xf32>, tensor<33x63x44x82x48x12xf32>) -> tensor<33x63x44x82x48x12xf32>
    %4 = tosa.tanh %3 : (tensor<33x63x44x82x48x12xf32>) -> tensor<33x63x44x82x48x12xf32>
    %5 = tosa.rsqrt %4 : (tensor<33x63x44x82x48x12xf32>) -> tensor<33x63x44x82x48x12xf32>
    %6 = tosa.reduce_sum %2 {axis = 3 : i32} : (tensor<1x118x49x79xi64>) -> tensor<1x118x49x1xi64>
    %7 = tosa.reduce_max %6 {axis = 3 : i32} : (tensor<1x118x49x1xi64>) -> tensor<1x118x49x1xi64>
    %8 = tosa.logical_right_shift %6, %6 : (tensor<1x118x49x1xi64>, tensor<1x118x49x1xi64>) -> tensor<1x118x49x1xi64>
    %9 = tosa.log %1 : (tensor<33x63x44x82x48x12xf32>) -> tensor<33x63x44x82x48x12xf32>
    %10 = tosa.reduce_any %arg4 {axis = 0 : i32} : (tensor<80xi1>) -> tensor<1xi1>
    %11 = tosa.logical_left_shift %10, %10 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.bitwise_not %10 : (tensor<1xi1>) -> tensor<1xi1>
    %13 = tosa.floor %5 : (tensor<33x63x44x82x48x12xf32>) -> tensor<33x63x44x82x48x12xf32>
    %14 = tosa.reduce_any %12 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %15 = tosa.reverse %12 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %16 = tosa.reduce_all %15 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %7, %8, %9, %11, %13, %14, %16 : tensor<1x118x49x1xi64>, tensor<1x118x49x1xi64>, tensor<33x63x44x82x48x12xf32>, tensor<1xi1>, tensor<33x63x44x82x48x12xf32>, tensor<1xi1>, tensor<1xi1>
  }
}
