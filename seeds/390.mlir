module {
  func.func @main(%arg0: tensor<83x14xi8>, %arg1: tensor<83x1xi8>, %arg2: tensor<37x49x29x17x52xf32>, %arg3: tensor<60x26x97x80x31x74xi1>, %arg4: tensor<1x1x1x1x1x74xi1>) -> (tensor<37x49x29x17x52xf32>, tensor<83x28xi8>, tensor<60x26x97x80x31x74xi1>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<83x14xi8>, tensor<83x1xi8>) -> tensor<83x14xi8>
    %1 = tosa.bitwise_and %0, %0 : (tensor<83x14xi8>, tensor<83x14xi8>) -> tensor<83x14xi8>
    %2 = tosa.tanh %arg2 : (tensor<37x49x29x17x52xf32>) -> tensor<37x49x29x17x52xf32>
    %3 = tosa.logical_right_shift %1, %1 : (tensor<83x14xi8>, tensor<83x14xi8>) -> tensor<83x14xi8>
    %4 = tosa.reverse %3 {axis = 0 : i32} : (tensor<83x14xi8>) -> tensor<83x14xi8>
    %5 = tosa.concat %4, %1 {axis = 1 : i32} : (tensor<83x14xi8>, tensor<83x14xi8>) -> tensor<83x28xi8>
    %6 = tosa.logical_left_shift %5, %5 : (tensor<83x28xi8>, tensor<83x28xi8>) -> tensor<83x28xi8>
    %7 = tosa.logical_xor %arg3, %arg4 : (tensor<60x26x97x80x31x74xi1>, tensor<1x1x1x1x1x74xi1>) -> tensor<60x26x97x80x31x74xi1>
    return %2, %6, %7 : tensor<37x49x29x17x52xf32>, tensor<83x28xi8>, tensor<60x26x97x80x31x74xi1>
  }
}
