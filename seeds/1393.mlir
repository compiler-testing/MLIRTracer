module {
  func.func @main(%arg0: tensor<38x93xi1>, %arg1: tensor<90xi8>, %arg2: tensor<1xi8>, %arg3: tensor<39xf32>, %arg4: tensor<39xf32>) -> (tensor<38x1xi1>, tensor<90xi1>, tensor<39xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<38x93xi1>) -> tensor<38x1xi1>
    %1 = tosa.greater_equal %arg1, %arg2 : (tensor<90xi8>, tensor<1xi8>) -> tensor<90xi1>
    %2 = tosa.equal %arg3, %arg4 : (tensor<39xf32>, tensor<39xf32>) -> tensor<39xi1>
    return %0, %1, %2 : tensor<38x1xi1>, tensor<90xi1>, tensor<39xi1>
  }
}
