module {
  func.func @main(%arg0: tensor<17x8x73x93xi1>, %arg1: tensor<42x63x88x98xi8>, %arg2: tensor<1x1x1x1xi8>, %arg3: tensor<18x18xf32>) -> (tensor<42x63x88x98xi1>, tensor<1x8x1x93xi1>, tensor<42x63x88x98xi1>, tensor<126x63x176x196xi1>, tensor<18x18xf32>) {
    %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<17x8x73x93xi1>) -> tensor<1x8x73x93xi1>
    %1 = tosa.maximum %arg1, %arg2 : (tensor<42x63x88x98xi8>, tensor<1x1x1x1xi8>) -> tensor<42x63x88x98xi8>
    %2 = tosa.identity %1 : (tensor<42x63x88x98xi8>) -> tensor<42x63x88x98xi8>
    %3 = tosa.ceil %arg3 : (tensor<18x18xf32>) -> tensor<18x18xf32>
    %4 = tosa.ceil %3 : (tensor<18x18xf32>) -> tensor<18x18xf32>
    %5 = tosa.greater %2, %1 : (tensor<42x63x88x98xi8>, tensor<42x63x88x98xi8>) -> tensor<42x63x88x98xi1>
    %6 = tosa.abs %2 : (tensor<42x63x88x98xi8>) -> tensor<42x63x88x98xi8>
    %7 = tosa.reverse %2 {axis = 3 : i32} : (tensor<42x63x88x98xi8>) -> tensor<42x63x88x98xi8>
    %8 = tosa.exp %4 : (tensor<18x18xf32>) -> tensor<18x18xf32>
    %9 = tosa.reduce_any %0 {axis = 2 : i32} : (tensor<1x8x73x93xi1>) -> tensor<1x8x1x93xi1>
    %10 = tosa.sub %8, %3 : (tensor<18x18xf32>, tensor<18x18xf32>) -> tensor<18x18xf32>
    %11 = tosa.add %7, %7 : (tensor<42x63x88x98xi8>, tensor<42x63x88x98xi8>) -> tensor<42x63x88x98xi8>
    %12 = tosa.logical_xor %9, %9 : (tensor<1x8x1x93xi1>, tensor<1x8x1x93xi1>) -> tensor<1x8x1x93xi1>
    %13 = tosa.tanh %10 : (tensor<18x18xf32>) -> tensor<18x18xf32>
    %14 = tosa.equal %7, %11 : (tensor<42x63x88x98xi8>, tensor<42x63x88x98xi8>) -> tensor<42x63x88x98xi1>
    %15 = tosa.equal %7, %6 : (tensor<42x63x88x98xi8>, tensor<42x63x88x98xi8>) -> tensor<42x63x88x98xi1>
    %t_16 = tosa.const_shape {values = dense<[ 3, 1, 2, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %16 = tosa.tile %15, %t_16 : (tensor<42x63x88x98xi1>, !tosa.shape<4>) -> tensor<126x63x176x196xi1>
    %17 = tosa.maximum %13, %10 : (tensor<18x18xf32>, tensor<18x18xf32>) -> tensor<18x18xf32>
    %t_18 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %18 = tosa.tile %17, %t_18 : (tensor<18x18xf32>, !tosa.shape<2>) -> tensor<18x18xf32>
    return %5, %12, %14, %16, %18 : tensor<42x63x88x98xi1>, tensor<1x8x1x93xi1>, tensor<42x63x88x98xi1>, tensor<126x63x176x196xi1>, tensor<18x18xf32>
  }
}
