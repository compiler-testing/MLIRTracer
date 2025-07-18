module {
  func.func @main(%arg0: tensor<79xf32>, %arg1: tensor<79xf32>, %arg2: tensor<f32>) -> (tensor<237xi1>, tensor<i1>) {
    %0 = tosa.equal %arg0, %arg1 : (tensor<79xf32>, tensor<79xf32>) -> tensor<79xi1>
    %1 = tosa.floor %arg2 : (tensor<f32>) -> tensor<f32>
    %t_2 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %2 = tosa.tile %0, %t_2 : (tensor<79xi1>, !tosa.shape<1>) -> tensor<237xi1>
    %3 = tosa.equal %1, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %2, %3 : tensor<237xi1>, tensor<i1>
  }
}
