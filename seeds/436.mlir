module {
  func.func @main(%arg0: tensor<39xi1>, %arg1: tensor<67x73x5x68xf32>) -> (tensor<39xi1>, tensor<67x73x5x68xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 1 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.tile %arg0, %t_0 : (tensor<39xi1>, !tosa.shape<1>) -> tensor<39xi1>
    %1 = tosa.ceil %arg1 : (tensor<67x73x5x68xf32>) -> tensor<67x73x5x68xf32>
    return %0, %1 : tensor<39xi1>, tensor<67x73x5x68xf32>
  }
}
