module {
  func.func @main(%arg0: tensor<10x18x84xi8>, %arg1: tensor<17xi1>, %arg2: tensor<70x80x6x28x86x59xf32>) -> (tensor<1x18x84xi8>, tensor<70x80x6x28x86x59xi1>, tensor<70x80x6x28x86x59xf32>, tensor<70x80x6x28x86x59xf32>, tensor<1xi1>, tensor<i32>, tensor<70x80x6x28x86x59xf32>, tensor<1xi1>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<10x18x84xi8>) -> tensor<1x18x84xi8>
    %1 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<17xi1>) -> tensor<1xi1>
    %2 = tosa.logical_not %1 : (tensor<1xi1>) -> tensor<1xi1>
    %3 = tosa.log %arg2 : (tensor<70x80x6x28x86x59xf32>) -> tensor<70x80x6x28x86x59xf32>
    %4 = tosa.greater_equal %3, %3 : (tensor<70x80x6x28x86x59xf32>, tensor<70x80x6x28x86x59xf32>) -> tensor<70x80x6x28x86x59xi1>
    %5 = tosa.log %3 : (tensor<70x80x6x28x86x59xf32>) -> tensor<70x80x6x28x86x59xf32>
    %6 = tosa.exp %5 : (tensor<70x80x6x28x86x59xf32>) -> tensor<70x80x6x28x86x59xf32>
    %7 = tosa.logical_right_shift %1, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %8 = tosa.abs %1 : (tensor<1xi1>) -> tensor<1xi1>
    %9 = tosa.exp %5 : (tensor<70x80x6x28x86x59xf32>) -> tensor<70x80x6x28x86x59xf32>
    %10 = tosa.reduce_max %8 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %11 = tosa.sub %8, %10 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %12 = tosa.logical_left_shift %7, %8 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %13 = tosa.argmax %8 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %14 = tosa.sigmoid %5 : (tensor<70x80x6x28x86x59xf32>) -> tensor<70x80x6x28x86x59xf32>
    %15 = tosa.reduce_any %12 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %0, %4, %6, %9, %11, %13, %14, %15 : tensor<1x18x84xi8>, tensor<70x80x6x28x86x59xi1>, tensor<70x80x6x28x86x59xf32>, tensor<70x80x6x28x86x59xf32>, tensor<1xi1>, tensor<i32>, tensor<70x80x6x28x86x59xf32>, tensor<1xi1>
  }
}
