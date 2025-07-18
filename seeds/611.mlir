module {
  func.func @main(%arg0: tensor<33x22x89xi64>, %arg1: tensor<17x99x20x16x4x66xf32>) -> (tensor<1x2xi64>, tensor<17x99x20x16x4x66xf32>, tensor<17x99x20x16x4x66xf32>, tensor<17x99x20x16x4x66xi1>, tensor<17x99x20x16x4x66xf32>, tensor<17x99x20x16x4x66xf32>) {
    %r_0 = tosa.const_shape {values = dense<[ 66, 979 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<33x22x89xi64>, !tosa.shape<2>) -> tensor<66x979xi64>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<66x979xi64>) -> tensor<66x1xi64>
    %t_2 = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.tile %1, %t_2 : (tensor<66x1xi64>, !tosa.shape<2>) -> tensor<66x2xi64>
    %3 = tosa.rsqrt %arg1 : (tensor<17x99x20x16x4x66xf32>) -> tensor<17x99x20x16x4x66xf32>
    %4 = tosa.greater_equal %3, %3 : (tensor<17x99x20x16x4x66xf32>, tensor<17x99x20x16x4x66xf32>) -> tensor<17x99x20x16x4x66xi1>
    %5 = tosa.logical_xor %4, %4 : (tensor<17x99x20x16x4x66xi1>, tensor<17x99x20x16x4x66xi1>) -> tensor<17x99x20x16x4x66xi1>
    %6 = tosa.equal %3, %3 : (tensor<17x99x20x16x4x66xf32>, tensor<17x99x20x16x4x66xf32>) -> tensor<17x99x20x16x4x66xi1>
    %7 = tosa.logical_and %5, %6 : (tensor<17x99x20x16x4x66xi1>, tensor<17x99x20x16x4x66xi1>) -> tensor<17x99x20x16x4x66xi1>
    %8 = tosa.ceil %3 : (tensor<17x99x20x16x4x66xf32>) -> tensor<17x99x20x16x4x66xf32>
    %9 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<66x2xi64>) -> tensor<1x2xi64>
    %10 = tosa.logical_right_shift %7, %4 : (tensor<17x99x20x16x4x66xi1>, tensor<17x99x20x16x4x66xi1>) -> tensor<17x99x20x16x4x66xi1>
    %11 = tosa.ceil %3 : (tensor<17x99x20x16x4x66xf32>) -> tensor<17x99x20x16x4x66xf32>
    %12 = tosa.floor %3 : (tensor<17x99x20x16x4x66xf32>) -> tensor<17x99x20x16x4x66xf32>
    %13 = tosa.bitwise_and %10, %4 : (tensor<17x99x20x16x4x66xi1>, tensor<17x99x20x16x4x66xi1>) -> tensor<17x99x20x16x4x66xi1>
    %14 = tosa.exp %8 : (tensor<17x99x20x16x4x66xf32>) -> tensor<17x99x20x16x4x66xf32>
    %15 = tosa.exp %3 : (tensor<17x99x20x16x4x66xf32>) -> tensor<17x99x20x16x4x66xf32>
    return %9, %11, %12, %13, %14, %15 : tensor<1x2xi64>, tensor<17x99x20x16x4x66xf32>, tensor<17x99x20x16x4x66xf32>, tensor<17x99x20x16x4x66xi1>, tensor<17x99x20x16x4x66xf32>, tensor<17x99x20x16x4x66xf32>
  }
}
