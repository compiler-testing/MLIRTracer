module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<9xi1>, %arg3: tensor<49xi1>, %arg4: tensor<60x30x8x4x42x40xf32>, %arg5: tensor<60x30x8x4x1x1xf32>) -> (tensor<i1>, tensor<i32>, tensor<1xi1>, tensor<60x30x8x4x42x40xf32>, tensor<60x30x8x4x42x40xf32>, tensor<1xi1>, tensor<1xi1>, tensor<60x30x8x4x42x40xi1>, tensor<1xi1>, tensor<1xi1>) {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.concat %arg2, %arg3 {axis = 0 : i32} : (tensor<9xi1>, tensor<49xi1>) -> tensor<58xi1>
    %2 = tosa.concat %1, %1 {axis = 0 : i32} : (tensor<58xi1>, tensor<58xi1>) -> tensor<116xi1>
    %3 = tosa.pow %arg4, %arg5 : (tensor<60x30x8x4x42x40xf32>, tensor<60x30x8x4x1x1xf32>) -> tensor<60x30x8x4x42x40xf32>
    %4 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<58xi1>) -> tensor<1xi1>
    %5 = tosa.argmax %4 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<i32>
    %6 = tosa.bitwise_and %2, %2 : (tensor<116xi1>, tensor<116xi1>) -> tensor<116xi1>
    %7 = tosa.reduce_max %6 {axis = 0 : i32} : (tensor<116xi1>) -> tensor<1xi1>
    %8 = tosa.reduce_all %7 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %9 = tosa.reduce_product %8 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %10 = tosa.logical_xor %9, %4 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %11 = tosa.ceil %3 : (tensor<60x30x8x4x42x40xf32>) -> tensor<60x30x8x4x42x40xf32>
    %12 = tosa.minimum %3, %3 : (tensor<60x30x8x4x42x40xf32>, tensor<60x30x8x4x42x40xf32>) -> tensor<60x30x8x4x42x40xf32>
    %13 = tosa.logical_or %9, %8 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %14 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<58xi1>) -> tensor<1xi1>
    %15 = tosa.logical_and %14, %9 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %16 = tosa.logical_right_shift %7, %8 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %17 = tosa.greater %3, %3 : (tensor<60x30x8x4x42x40xf32>, tensor<60x30x8x4x42x40xf32>) -> tensor<60x30x8x4x42x40xi1>
    %18 = tosa.reduce_max %13 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %19 = tosa.reduce_sum %14 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %0, %5, %10, %11, %12, %15, %16, %17, %18, %19 : tensor<i1>, tensor<i32>, tensor<1xi1>, tensor<60x30x8x4x42x40xf32>, tensor<60x30x8x4x42x40xf32>, tensor<1xi1>, tensor<1xi1>, tensor<60x30x8x4x42x40xi1>, tensor<1xi1>, tensor<1xi1>
  }
}
