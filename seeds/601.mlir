module {
  func.func @main(%arg0: tensor<51x71x54x59xf32>, %arg1: tensor<1x1x54x1xf32>, %arg2: tensor<64x19x96x38x9xi32>, %arg3: tensor<64x19x96x1x9xi32>, %arg4: tensor<69x68x9xi1>, %arg5: tensor<1x68x9xi1>) -> (tensor<69x68x9xi1>, tensor<64x19x96x38x9xi32>, tensor<69x68x9xi1>, tensor<69x204x9xi1>, tensor<69x204x1xi1>, tensor<51x71x54x59xi1>, tensor<51x71x54x59xf32>, tensor<51x71x54x59xf32>, tensor<51x71x54x59xf32>, tensor<69x204x9xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<51x71x54x59xf32>, tensor<1x1x54x1xf32>) -> tensor<51x71x54x59xf32>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<64x19x96x38x9xi32>, tensor<64x19x96x1x9xi32>) -> tensor<64x19x96x38x9xi32>
    %2 = tosa.reciprocal %0 : (tensor<51x71x54x59xf32>) -> tensor<51x71x54x59xf32>
    %3 = tosa.sub %1, %1 : (tensor<64x19x96x38x9xi32>, tensor<64x19x96x38x9xi32>) -> tensor<64x19x96x38x9xi32>
    %4 = tosa.identity %2 : (tensor<51x71x54x59xf32>) -> tensor<51x71x54x59xf32>
    %5 = tosa.reciprocal %4 : (tensor<51x71x54x59xf32>) -> tensor<51x71x54x59xf32>
    %6 = tosa.minimum %3, %3 : (tensor<64x19x96x38x9xi32>, tensor<64x19x96x38x9xi32>) -> tensor<64x19x96x38x9xi32>
    %7 = tosa.logical_or %arg4, %arg5 : (tensor<69x68x9xi1>, tensor<1x68x9xi1>) -> tensor<69x68x9xi1>
    %8 = tosa.greater %5, %4 : (tensor<51x71x54x59xf32>, tensor<51x71x54x59xf32>) -> tensor<51x71x54x59xi1>
    %9 = tosa.logical_xor %7, %7 : (tensor<69x68x9xi1>, tensor<69x68x9xi1>) -> tensor<69x68x9xi1>
    %10 = tosa.intdiv %1, %6 : (tensor<64x19x96x38x9xi32>, tensor<64x19x96x38x9xi32>) -> tensor<64x19x96x38x9xi32>
    %11 = tosa.logical_left_shift %7, %7 : (tensor<69x68x9xi1>, tensor<69x68x9xi1>) -> tensor<69x68x9xi1>
    %12 = tosa.abs %8 : (tensor<51x71x54x59xi1>) -> tensor<51x71x54x59xi1>
    %t_13 = tosa.const_shape {values = dense<[ 1, 3, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %13 = tosa.tile %7, %t_13 : (tensor<69x68x9xi1>, !tosa.shape<3>) -> tensor<69x204x9xi1>
    %14 = tosa.add %13, %13 : (tensor<69x204x9xi1>, tensor<69x204x9xi1>) -> tensor<69x204x9xi1>
    %15 = tosa.reduce_max %13 {axis = 2 : i32} : (tensor<69x204x9xi1>) -> tensor<69x204x1xi1>
    %16 = tosa.reverse %12 {axis = 1 : i32} : (tensor<51x71x54x59xi1>) -> tensor<51x71x54x59xi1>
    %17 = tosa.exp %2 : (tensor<51x71x54x59xf32>) -> tensor<51x71x54x59xf32>
    %18 = tosa.minimum %0, %0 : (tensor<51x71x54x59xf32>, tensor<51x71x54x59xf32>) -> tensor<51x71x54x59xf32>
    %19 = tosa.tanh %2 : (tensor<51x71x54x59xf32>) -> tensor<51x71x54x59xf32>
    %20 = tosa.bitwise_xor %13, %13 : (tensor<69x204x9xi1>, tensor<69x204x9xi1>) -> tensor<69x204x9xi1>
    return %9, %10, %11, %14, %15, %16, %17, %18, %19, %20 : tensor<69x68x9xi1>, tensor<64x19x96x38x9xi32>, tensor<69x68x9xi1>, tensor<69x204x9xi1>, tensor<69x204x1xi1>, tensor<51x71x54x59xi1>, tensor<51x71x54x59xf32>, tensor<51x71x54x59xf32>, tensor<51x71x54x59xf32>, tensor<69x204x9xi1>
  }
}
