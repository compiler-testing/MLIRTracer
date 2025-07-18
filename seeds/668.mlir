module {
  func.func @main(%arg0: tensor<37x12xi8>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<12x1x58x88xf32>) -> (tensor<74x12xi8>, tensor<i32>, tensor<1x1x58x88xf32>) {
    %0 = tosa.abs %arg0 : (tensor<37x12xi8>) -> tensor<37x12xi8>
    %1 = tosa.minimum %0, %0 : (tensor<37x12xi8>, tensor<37x12xi8>) -> tensor<37x12xi8>
    %2 = tosa.sub %1, %0 : (tensor<37x12xi8>, tensor<37x12xi8>) -> tensor<37x12xi8>
    %t_3 = tosa.const_shape {values = dense<[ 2, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %2, %t_3 : (tensor<37x12xi8>, !tosa.shape<2>) -> tensor<74x12xi8>
    %4 = tosa.intdiv %arg1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = tosa.log %arg3 : (tensor<12x1x58x88xf32>) -> tensor<12x1x58x88xf32>
    %6 = tosa.reduce_max %5 {axis = 0 : i32} : (tensor<12x1x58x88xf32>) -> tensor<1x1x58x88xf32>
    return %3, %4, %6 : tensor<74x12xi8>, tensor<i32>, tensor<1x1x58x88xf32>
  }
}
