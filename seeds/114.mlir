module {
  func.func @main(%arg0: tensor<82x17x43xi32>, %arg1: tensor<50x22x28x89x79xf32>) -> (tensor<82x17x43xi32>, tensor<50x22x28x89x79xf32>) {
    %0 = tosa.reverse %arg0 {axis = 2 : i32} : (tensor<82x17x43xi32>) -> tensor<82x17x43xi32>
    %1 = tosa.exp %arg1 : (tensor<50x22x28x89x79xf32>) -> tensor<50x22x28x89x79xf32>
    return %0, %1 : tensor<82x17x43xi32>, tensor<50x22x28x89x79xf32>
  }
}
