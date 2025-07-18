module {
  func.func @main(%arg0: tensor<89x56x71x22x75x91xf32>, %arg1: tensor<55x75x13x14x19x31xi1>, %arg2: tensor<78x23xi1>) -> (tensor<10x2x6x3x1x11xi1>, tensor<89x56x71x22x75x91xi1>, tensor<78x1xi1>) {
    %0 = tosa.log %arg0 : (tensor<89x56x71x22x75x91xf32>) -> tensor<89x56x71x22x75x91xf32>
    %1 = tosa.floor %0 : (tensor<89x56x71x22x75x91xf32>) -> tensor<89x56x71x22x75x91xf32>
    %2 = tosa.logical_not %arg1 : (tensor<55x75x13x14x19x31xi1>) -> tensor<55x75x13x14x19x31xi1>
    %3 = tosa.reduce_all %arg2 {axis = 1 : i32} : (tensor<78x23xi1>) -> tensor<78x1xi1>
    %s_4_start = tosa.const_shape {values = dense<[ 14, 35, 3, 11, 8, 20 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %s_4_size = tosa.const_shape {values = dense<[ 10, 2, 6, 3, 1, 11 ]> : tensor<6xindex>} : () -> !tosa.shape<6>
    %4 = tosa.slice %2, %s_4_start, %s_4_size : (tensor<55x75x13x14x19x31xi1>, !tosa.shape<6>, !tosa.shape<6>) -> tensor<10x2x6x3x1x11xi1>
    %5 = tosa.equal %1, %0 : (tensor<89x56x71x22x75x91xf32>, tensor<89x56x71x22x75x91xf32>) -> tensor<89x56x71x22x75x91xi1>
    %6 = tosa.logical_xor %5, %5 : (tensor<89x56x71x22x75x91xi1>, tensor<89x56x71x22x75x91xi1>) -> tensor<89x56x71x22x75x91xi1>
    %7 = tosa.arithmetic_right_shift %6, %6 {round = true} : (tensor<89x56x71x22x75x91xi1>, tensor<89x56x71x22x75x91xi1>) -> tensor<89x56x71x22x75x91xi1>
    %8 = tosa.reduce_all %3 {axis = 1 : i32} : (tensor<78x1xi1>) -> tensor<78x1xi1>
    return %4, %7, %8 : tensor<10x2x6x3x1x11xi1>, tensor<89x56x71x22x75x91xi1>, tensor<78x1xi1>
  }
}
