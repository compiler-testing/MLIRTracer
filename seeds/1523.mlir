module {
  func.func @main(%arg0: tensor<69x28xi1>, %arg1: tensor<74x97x40x55x90x23xi64>, %arg2: tensor<1x1x1x1x90x1xi64>, %arg3: tensor<83x7x3x93xi64>, %arg4: tensor<83x1x1x1xi64>, %arg5: tensor<11xi32>, %arg6: tensor<11xi32>, %arg7: tensor<86x68x32xf32>) -> (tensor<74x97x40x55x90x23xi1>, tensor<1x1xi1>, tensor<83x7x3x93xi1>, tensor<86x68x32xf32>, tensor<1xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<69x28xi1>) -> tensor<1x28xi1>
    %1 = tosa.greater %arg1, %arg2 : (tensor<74x97x40x55x90x23xi64>, tensor<1x1x1x1x90x1xi64>) -> tensor<74x97x40x55x90x23xi1>
    %2 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<1x28xi1>) -> tensor<1x1xi1>
    %3 = tosa.sub %1, %1 : (tensor<74x97x40x55x90x23xi1>, tensor<74x97x40x55x90x23xi1>) -> tensor<74x97x40x55x90x23xi1>
    %4 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<1x1xi1>) -> tensor<1x1xi1>
    %5 = tosa.equal %arg3, %arg4 : (tensor<83x7x3x93xi64>, tensor<83x1x1x1xi64>) -> tensor<83x7x3x93xi1>
    %6 = tosa.intdiv %arg5, %arg6 : (tensor<11xi32>, tensor<11xi32>) -> tensor<11xi32>
    %7 = tosa.add %5, %5 : (tensor<83x7x3x93xi1>, tensor<83x7x3x93xi1>) -> tensor<83x7x3x93xi1>
    %8 = tosa.reduce_sum %6 {axis = 0 : i32} : (tensor<11xi32>) -> tensor<1xi32>
    %9 = tosa.clamp %8 {min_val = 7 : i32, max_val = 134 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %10 = tosa.floor %arg7 : (tensor<86x68x32xf32>) -> tensor<86x68x32xf32>
    %11 = tosa.equal %9, %8 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    return %3, %4, %7, %10, %11 : tensor<74x97x40x55x90x23xi1>, tensor<1x1xi1>, tensor<83x7x3x93xi1>, tensor<86x68x32xf32>, tensor<1xi1>
  }
}
