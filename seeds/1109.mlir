module {
  func.func @main(%arg0: tensor<15x89x73x39xi8>, %arg1: tensor<87xf32>, %arg2: tensor<59x64xi1>) -> (tensor<15x1x73x39xi8>, tensor<87xf32>, tensor<1xf32>, tensor<59x64xi1>) {
    %0 = tosa.reduce_max %arg0 {axis = 1 : i32} : (tensor<15x89x73x39xi8>) -> tensor<15x1x73x39xi8>
    %1 = tosa.exp %arg1 : (tensor<87xf32>) -> tensor<87xf32>
    %2 = tosa.add %1, %1 : (tensor<87xf32>, tensor<87xf32>) -> tensor<87xf32>
    %3 = tosa.ceil %2 : (tensor<87xf32>) -> tensor<87xf32>
    %4 = tosa.floor %3 : (tensor<87xf32>) -> tensor<87xf32>
    %5 = tosa.log %2 : (tensor<87xf32>) -> tensor<87xf32>
    %6 = tosa.floor %5 : (tensor<87xf32>) -> tensor<87xf32>
    %t_7 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %7 = tosa.tile %6, %t_7 : (tensor<87xf32>, !tosa.shape<1>) -> tensor<87xf32>
    %8 = tosa.reduce_max %4 {axis = 0 : i32} : (tensor<87xf32>) -> tensor<1xf32>
    %9 = tosa.logical_not %arg2 : (tensor<59x64xi1>) -> tensor<59x64xi1>
    return %0, %7, %8, %9 : tensor<15x1x73x39xi8>, tensor<87xf32>, tensor<1xf32>, tensor<59x64xi1>
  }
}
