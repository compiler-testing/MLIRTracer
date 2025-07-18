module {
  func.func @main(%arg0: tensor<61x64x58xf32>) -> (tensor<3x22x12xi1>, tensor<61x64x58xf32>) {
    %0 = tosa.identity %arg0 : (tensor<61x64x58xf32>) -> tensor<61x64x58xf32>
    %1 = tosa.greater_equal %0, %0 : (tensor<61x64x58xf32>, tensor<61x64x58xf32>) -> tensor<61x64x58xi1>
    %2 = tosa.arithmetic_right_shift %1, %1 {round = true} : (tensor<61x64x58xi1>, tensor<61x64x58xi1>) -> tensor<61x64x58xi1>
    %3 = tosa.reduce_any %2 {axis = 0 : i32} : (tensor<61x64x58xi1>) -> tensor<1x64x58xi1>
    %4 = tosa.logical_right_shift %3, %3 : (tensor<1x64x58xi1>, tensor<1x64x58xi1>) -> tensor<1x64x58xi1>
    %s_5_start = tosa.const_shape {values = dense<[ 0, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_5_size = tosa.const_shape {values = dense<[ 3, 11, 12 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<1x64x58xi1>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<3x11x12xi1>
    %6 = tosa.sub %5, %5 : (tensor<3x11x12xi1>, tensor<3x11x12xi1>) -> tensor<3x11x12xi1>
    %7 = tosa.concat %6, %6 {axis = 1 : i32} : (tensor<3x11x12xi1>, tensor<3x11x12xi1>) -> tensor<3x22x12xi1>
    %8 = tosa.ceil %0 : (tensor<61x64x58xf32>) -> tensor<61x64x58xf32>
    %9 = tosa.reciprocal %8 : (tensor<61x64x58xf32>) -> tensor<61x64x58xf32>
    %10 = tosa.reciprocal %9 : (tensor<61x64x58xf32>) -> tensor<61x64x58xf32>
    return %7, %10 : tensor<3x22x12xi1>, tensor<61x64x58xf32>
  }
}
