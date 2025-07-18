module {
  func.func @main(%arg0: tensor<97xi16>, %arg1: tensor<58x45x98xi32>, %arg2: tensor<58x45x98xi32>, %arg3: tensor<12xf32>) -> (tensor<291xi16>, tensor<12xf32>, tensor<58x1x98xi1>) {
    %t_0 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<97xi16>, !tosa.shape<1>) -> tensor<291xi16>
    %1 = tosa.abs %0 : (tensor<291xi16>) -> tensor<291xi16>
    %2 = tosa.logical_right_shift %1, %1 : (tensor<291xi16>, tensor<291xi16>) -> tensor<291xi16>
    %3 = tosa.greater %arg1, %arg2 : (tensor<58x45x98xi32>, tensor<58x45x98xi32>) -> tensor<58x45x98xi1>
    %4 = tosa.reverse %2 {axis = 0 : i32} : (tensor<291xi16>) -> tensor<291xi16>
    %5 = tosa.clz %3 : (tensor<58x45x98xi1>) -> tensor<58x45x98xi1>
    %6 = tosa.exp %arg3 : (tensor<12xf32>) -> tensor<12xf32>
    %7 = tosa.clz %4 : (tensor<291xi16>) -> tensor<291xi16>
    %8 = tosa.floor %6 : (tensor<12xf32>) -> tensor<12xf32>
    %9 = tosa.reduce_max %5 {axis = 1 : i32} : (tensor<58x45x98xi1>) -> tensor<58x1x98xi1>
    return %7, %8, %9 : tensor<291xi16>, tensor<12xf32>, tensor<58x1x98xi1>
  }
}
