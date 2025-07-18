module {
  func.func @main(%arg0: tensor<34x7x46xi1>, %arg1: tensor<42x57x80xf32>, %arg2: tensor<42x57x80xf32>, %arg3: tensor<59x74x80x48x69x31xi8>, %arg4: tensor<59x74x1x1x1x31xi8>, %arg5: tensor<94x69x29x46x24x77xi32>, %arg6: tensor<1x1x29x1x24x1xi32>, %arg7: tensor<23x6x5x33x57xf32>) -> (tensor<34x7x46xi1>, tensor<59x74x80x48x138x31xi1>, tensor<94x69x29x46x24x77xi32>, tensor<94x69x29x46x24x77xi32>, tensor<59x74x80x48x69x31xi1>, tensor<42x80x1xi1>, tensor<126x240x114xi1>, tensor<23x6x10x33x57xf32>, tensor<23x6x5x33x57xf32>) {
    %0 = tosa.logical_not %arg0 : (tensor<34x7x46xi1>) -> tensor<34x7x46xi1>
    %1 = tosa.greater %arg1, %arg2 : (tensor<42x57x80xf32>, tensor<42x57x80xf32>) -> tensor<42x57x80xi1>
    %2 = tosa.sub %1, %1 : (tensor<42x57x80xi1>, tensor<42x57x80xi1>) -> tensor<42x57x80xi1>
    %3 = tosa.logical_right_shift %0, %0 : (tensor<34x7x46xi1>, tensor<34x7x46xi1>) -> tensor<34x7x46xi1>
    %4 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %5 = tosa.transpose %2 {perms = array<i32: 0, 2, 1>} : (tensor<42x57x80xi1>) -> tensor<42x80x57xi1>
    %6 = tosa.greater %arg3, %arg4 : (tensor<59x74x80x48x69x31xi8>, tensor<59x74x1x1x1x31xi8>) -> tensor<59x74x80x48x69x31xi1>
    %7 = tosa.minimum %arg5, %arg6 : (tensor<94x69x29x46x24x77xi32>, tensor<1x1x29x1x24x1xi32>) -> tensor<94x69x29x46x24x77xi32>
    %8 = tosa.reverse %3 {axis = 0 : i32} : (tensor<34x7x46xi1>) -> tensor<34x7x46xi1>
    %9 = tosa.reduce_max %5 {axis = 2 : i32} : (tensor<42x80x57xi1>) -> tensor<42x80x1xi1>
    %10 = tosa.concat %6, %6 {axis = 4 : i32} : (tensor<59x74x80x48x69x31xi1>, tensor<59x74x80x48x69x31xi1>) -> tensor<59x74x80x48x138x31xi1>
    %11 = tosa.maximum %7, %7 : (tensor<94x69x29x46x24x77xi32>, tensor<94x69x29x46x24x77xi32>) -> tensor<94x69x29x46x24x77xi32>
    %12 = tosa.abs %7 : (tensor<94x69x29x46x24x77xi32>) -> tensor<94x69x29x46x24x77xi32>
    %13 = tosa.bitwise_xor %6, %6 : (tensor<59x74x80x48x69x31xi1>, tensor<59x74x80x48x69x31xi1>) -> tensor<59x74x80x48x69x31xi1>
    %14 = tosa.sigmoid %arg7 : (tensor<23x6x5x33x57xf32>) -> tensor<23x6x5x33x57xf32>
    %15 = tosa.reverse %9 {axis = 2 : i32} : (tensor<42x80x1xi1>) -> tensor<42x80x1xi1>
    %t_16 = tosa.const_shape {values = dense<[ 3, 3, 2 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %16 = tosa.tile %5, %t_16 : (tensor<42x80x57xi1>, !tosa.shape<3>) -> tensor<126x240x114xi1>
    %17 = tosa.concat %14, %14 {axis = 2 : i32} : (tensor<23x6x5x33x57xf32>, tensor<23x6x5x33x57xf32>) -> tensor<23x6x10x33x57xf32>
    %18 = tosa.log %14 : (tensor<23x6x5x33x57xf32>) -> tensor<23x6x5x33x57xf32>
    return %8, %10, %11, %12, %13, %15, %16, %17, %18 : tensor<34x7x46xi1>, tensor<59x74x80x48x138x31xi1>, tensor<94x69x29x46x24x77xi32>, tensor<94x69x29x46x24x77xi32>, tensor<59x74x80x48x69x31xi1>, tensor<42x80x1xi1>, tensor<126x240x114xi1>, tensor<23x6x10x33x57xf32>, tensor<23x6x5x33x57xf32>
  }
}
