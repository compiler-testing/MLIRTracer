module {
  func.func @main(%arg0: tensor<2x5x11x41x84xi1>, %arg1: tensor<1x1x1x41x84xi1>, %arg2: tensor<22x18xi8>, %arg3: tensor<1x1xi8>, %arg4: tensor<40x53x62xi64>, %arg5: tensor<40x53x1xi64>, %arg6: tensor<57x53x87x48x22x25xi32>, %arg7: tensor<57x1x87x1x22x25xi32>, %arg8: tensor<50xi8>, %arg9: tensor<1xi8>, %arg10: tensor<20x74x16x24x26x86xi64>, %arg11: tensor<1x1x16x24x1x1xi64>, %arg12: tensor<4x44x85x43x44xf32>) -> (tensor<2x5x11x41x84xi1>, tensor<40x53x62xi1>, tensor<22x18xi1>, tensor<22x18xi1>, tensor<57x53x87x48x22x25xi1>, tensor<50xi1>, tensor<20x74x16x24x26x86xi1>, tensor<20x74x16x24x26x86xi1>, tensor<i32>, tensor<4x44x85x43x44xf32>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<2x5x11x41x84xi1>, tensor<1x1x1x41x84xi1>) -> tensor<2x5x11x41x84xi1>
    %1 = tosa.equal %arg2, %arg3 : (tensor<22x18xi8>, tensor<1x1xi8>) -> tensor<22x18xi1>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<22x18xi1>) -> tensor<1x18xi1>
    %3 = tosa.equal %arg4, %arg5 : (tensor<40x53x62xi64>, tensor<40x53x1xi64>) -> tensor<40x53x62xi1>
    %4 = tosa.clz %1 : (tensor<22x18xi1>) -> tensor<22x18xi1>
    %5 = tosa.logical_and %1, %1 : (tensor<22x18xi1>, tensor<22x18xi1>) -> tensor<22x18xi1>
    %r_6 = tosa.const_shape {values = dense<[ 18 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.reshape %2, %r_6 : (tensor<1x18xi1>, !tosa.shape<1>) -> tensor<18xi1>
    %7 = tosa.greater %arg6, %arg7 : (tensor<57x53x87x48x22x25xi32>, tensor<57x1x87x1x22x25xi32>) -> tensor<57x53x87x48x22x25xi1>
    %8 = tosa.greater %arg8, %arg9 : (tensor<50xi8>, tensor<1xi8>) -> tensor<50xi1>
    %9 = tosa.maximum %arg10, %arg11 : (tensor<20x74x16x24x26x86xi64>, tensor<1x1x16x24x1x1xi64>) -> tensor<20x74x16x24x26x86xi64>
    %10 = tosa.argmax %6 {axis = 0 : i32} : (tensor<18xi1>) -> tensor<i32>
    %11 = tosa.intdiv %10, %10 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %12 = tosa.greater_equal %9, %9 : (tensor<20x74x16x24x26x86xi64>, tensor<20x74x16x24x26x86xi64>) -> tensor<20x74x16x24x26x86xi1>
    %13 = tosa.exp %arg12 : (tensor<4x44x85x43x44xf32>) -> tensor<4x44x85x43x44xf32>
    %14 = tosa.greater %9, %9 : (tensor<20x74x16x24x26x86xi64>, tensor<20x74x16x24x26x86xi64>) -> tensor<20x74x16x24x26x86xi1>
    %15 = tosa.intdiv %11, %11 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %16 = tosa.reciprocal %13 : (tensor<4x44x85x43x44xf32>) -> tensor<4x44x85x43x44xf32>
    return %0, %3, %4, %5, %7, %8, %12, %14, %15, %16 : tensor<2x5x11x41x84xi1>, tensor<40x53x62xi1>, tensor<22x18xi1>, tensor<22x18xi1>, tensor<57x53x87x48x22x25xi1>, tensor<50xi1>, tensor<20x74x16x24x26x86xi1>, tensor<20x74x16x24x26x86xi1>, tensor<i32>, tensor<4x44x85x43x44xf32>
  }
}
