module {
  func.func @main(%arg0: tensor<43xi16>, %arg1: tensor<43xi16>, %arg2: tensor<i1>, %arg3: tensor<i1>, %arg4: tensor<30x4xf32>) -> (tensor<1xi16>, tensor<i1>, tensor<30x4xf32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<43xi16>, tensor<43xi16>) -> tensor<43xi16>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<43xi16>) -> tensor<1xi16>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.sigmoid %arg4 : (tensor<30x4xf32>) -> tensor<30x4xf32>
    return %1, %2, %3 : tensor<1xi16>, tensor<i1>, tensor<30x4xf32>
  }
}
