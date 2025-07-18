module {
  func.func @main(%arg0: tensor<57x26x55xi64>, %arg1: tensor<1x26x1xi64>, %arg2: tensor<62x21xf32>) -> (tensor<57x1x1xi1>, tensor<2xi32>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<57x26x55xi64>, tensor<1x26x1xi64>) -> tensor<57x26x55xi1>
    %1 = tosa.concat %0, %0 {axis = 2 : i32} : (tensor<57x26x55xi1>, tensor<57x26x55xi1>) -> tensor<57x26x110xi1>
    %2 = tosa.reduce_any %1 {axis = 2 : i32} : (tensor<57x26x110xi1>) -> tensor<57x26x1xi1>
    %3 = tosa.add %2, %2 : (tensor<57x26x1xi1>, tensor<57x26x1xi1>) -> tensor<57x26x1xi1>
    %4 = tosa.reduce_min %3 {axis = 2 : i32} : (tensor<57x26x1xi1>) -> tensor<57x26x1xi1>
    %5 = tosa.bitwise_xor %4, %2 : (tensor<57x26x1xi1>, tensor<57x26x1xi1>) -> tensor<57x26x1xi1>
    %6 = tosa.ceil %arg2 : (tensor<62x21xf32>) -> tensor<62x21xf32>
    %7 = tosa.reduce_product %6 {axis = 1 : i32} : (tensor<62x21xf32>) -> tensor<62x1xf32>
    %8 = tosa.logical_left_shift %5, %5 : (tensor<57x26x1xi1>, tensor<57x26x1xi1>) -> tensor<57x26x1xi1>
    %9 = tosa.logical_right_shift %8, %8 : (tensor<57x26x1xi1>, tensor<57x26x1xi1>) -> tensor<57x26x1xi1>
    %10 = tosa.bitwise_xor %9, %4 : (tensor<57x26x1xi1>, tensor<57x26x1xi1>) -> tensor<57x26x1xi1>
    %11 = tosa.reduce_max %10 {axis = 1 : i32} : (tensor<57x26x1xi1>) -> tensor<57x1x1xi1>
    %12 = tosa.argmax %7 {axis = 0 : i32} : (tensor<62x1xf32>) -> tensor<1xi32>
    %13 = tosa.minimum %12, %12 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %14 = tosa.reverse %12 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %15 = tosa.concat %13, %14 {axis = 0 : i32} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    return %11, %15 : tensor<57x1x1xi1>, tensor<2xi32>
  }
}
