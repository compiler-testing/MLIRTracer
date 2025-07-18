module {
  func.func @main(%arg0: tensor<14x22x83xi32>, %arg1: tensor<1x1x1xi32>, %arg2: tensor<46x37x94x71xf32>, %arg3: tensor<i1>, %arg4: tensor<43x8x62x71xi1>) -> (tensor<i1>, tensor<46x37x94x71xf32>, tensor<43x8x62x1xi1>, tensor<581x1xi1>, tensor<46x37x94x71xf32>, tensor<9x5xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<14x22x83xi32>, tensor<1x1x1xi32>) -> tensor<14x22x83xi32>
    %1 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<14x22x83xi32>) -> tensor<14x1x83xi32>
    %r_2 = tosa.const_shape {values = dense<[ 581, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.reshape %1, %r_2 : (tensor<14x1x83xi32>, !tosa.shape<2>) -> tensor<581x2xi32>
    %3 = tosa.clz %2 : (tensor<581x2xi32>) -> tensor<581x2xi32>
    %4 = tosa.reduce_min %3 {axis = 1 : i32} : (tensor<581x2xi32>) -> tensor<581x1xi32>
    %5 = tosa.sigmoid %arg2 : (tensor<46x37x94x71xf32>) -> tensor<46x37x94x71xf32>
    %6 = tosa.bitwise_or %4, %4 : (tensor<581x1xi32>, tensor<581x1xi32>) -> tensor<581x1xi32>
    %7 = tosa.clamp %6 {min_val = 34 : i32, max_val = 119 : i32} : (tensor<581x1xi32>) -> tensor<581x1xi32>
    %8 = tosa.logical_not %arg3 : (tensor<i1>) -> tensor<i1>
    %9 = tosa.reduce_any %arg4 {axis = 3 : i32} : (tensor<43x8x62x71xi1>) -> tensor<43x8x62x1xi1>
    %10 = tosa.bitwise_and %8, %8 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %11 = tosa.clz %10 : (tensor<i1>) -> tensor<i1>
    %12 = tosa.log %5 : (tensor<46x37x94x71xf32>) -> tensor<46x37x94x71xf32>
    %13 = tosa.reciprocal %12 : (tensor<46x37x94x71xf32>) -> tensor<46x37x94x71xf32>
    %14 = tosa.sub %9, %9 : (tensor<43x8x62x1xi1>, tensor<43x8x62x1xi1>) -> tensor<43x8x62x1xi1>
    %15 = tosa.equal %7, %4 : (tensor<581x1xi32>, tensor<581x1xi32>) -> tensor<581x1xi1>
    %16 = tosa.equal %6, %4 : (tensor<581x1xi32>, tensor<581x1xi32>) -> tensor<581x1xi1>
    %17 = tosa.tanh %12 : (tensor<46x37x94x71xf32>) -> tensor<46x37x94x71xf32>
    %s_18_start = tosa.const_shape {values = dense<[ 463, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_18_size = tosa.const_shape {values = dense<[ 9, 5 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %18 = tosa.slice %15, %s_18_start, %s_18_size : (tensor<581x1xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<9x5xi1>
    return %11, %13, %14, %16, %17, %18 : tensor<i1>, tensor<46x37x94x71xf32>, tensor<43x8x62x1xi1>, tensor<581x1xi1>, tensor<46x37x94x71xf32>, tensor<9x5xi1>
  }
}
