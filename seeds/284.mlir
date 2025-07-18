module {
  func.func @main(%arg0: tensor<2x9x36x58x88xi32>, %arg1: tensor<2x9x1x58x1xi32>, %arg2: tensor<82xi32>, %arg3: tensor<1x74x86x88x73xf32>, %arg4: tensor<64xi1>) -> (tensor<2x9x36x58x88xi32>, tensor<1x74x86x88x73xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1x74x86x88x73xf32>, tensor<5xi1>, tensor<1x74x172x88x73xf32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<2x9x36x58x88xi32>, tensor<2x9x1x58x1xi32>) -> tensor<2x9x36x58x88xi32>
    %1 = tosa.maximum %0, %0 : (tensor<2x9x36x58x88xi32>, tensor<2x9x36x58x88xi32>) -> tensor<2x9x36x58x88xi32>
    %2 = tosa.clz %1 : (tensor<2x9x36x58x88xi32>) -> tensor<2x9x36x58x88xi32>
    %3 = tosa.intdiv %2, %1 : (tensor<2x9x36x58x88xi32>, tensor<2x9x36x58x88xi32>) -> tensor<2x9x36x58x88xi32>
    %4 = tosa.reduce_sum %arg2 {axis = 0 : i32} : (tensor<82xi32>) -> tensor<1xi32>
    %5 = tosa.sub %4, %4 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %6 = tosa.log %arg3 : (tensor<1x74x86x88x73xf32>) -> tensor<1x74x86x88x73xf32>
    %7 = tosa.reverse %5 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %8 = tosa.reduce_max %7 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %9 = tosa.reciprocal %6 : (tensor<1x74x86x88x73xf32>) -> tensor<1x74x86x88x73xf32>
    %10 = tosa.reciprocal %9 : (tensor<1x74x86x88x73xf32>) -> tensor<1x74x86x88x73xf32>
    %11 = tosa.clamp %10 {min_val = 6.000000e+00 : f32, max_val = 4.100000e+01 : f32} : (tensor<1x74x86x88x73xf32>) -> tensor<1x74x86x88x73xf32>
    %12 = tosa.reduce_all %arg4 {axis = 0 : i32} : (tensor<64xi1>) -> tensor<1xi1>
    %13 = tosa.reverse %5 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %14 = tosa.logical_right_shift %5, %13 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %15 = tosa.bitwise_or %13, %8 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %16 = tosa.clamp %6 {min_val = 6.000000e+00 : f32, max_val = 4.100000e+01 : f32} : (tensor<1x74x86x88x73xf32>) -> tensor<1x74x86x88x73xf32>
    %s_17_start = tosa.const_shape {values = dense<[ 0 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_17_size = tosa.const_shape {values = dense<[ 5 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %17 = tosa.slice %12, %s_17_start, %s_17_size : (tensor<1xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<5xi1>
    %18 = tosa.concat %6, %10 {axis = 2 : i32} : (tensor<1x74x86x88x73xf32>, tensor<1x74x86x88x73xf32>) -> tensor<1x74x172x88x73xf32>
    return %3, %11, %14, %15, %16, %17, %18 : tensor<2x9x36x58x88xi32>, tensor<1x74x86x88x73xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1x74x86x88x73xf32>, tensor<5xi1>, tensor<1x74x172x88x73xf32>
  }
}
