module {
  func.func @main(%arg0: tensor<88xf32>, %arg1: tensor<82xf32>) -> tensor<1020xf32> {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<88xf32>, tensor<82xf32>) -> tensor<170xf32>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<170xf32>, tensor<170xf32>) -> tensor<340xf32>
    %2 = tosa.sub %1, %1 : (tensor<340xf32>, tensor<340xf32>) -> tensor<340xf32>
    %3 = tosa.minimum %2, %2 : (tensor<340xf32>, tensor<340xf32>) -> tensor<340xf32>
    %t_4 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.tile %3, %t_4 : (tensor<340xf32>, !tosa.shape<1>) -> tensor<1020xf32>
    return %4 : tensor<1020xf32>
  }
}
