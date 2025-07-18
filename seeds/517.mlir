module {
  func.func @main(%arg0: tensor<67x74x64x42xf32>, %arg1: tensor<77xi1>, %arg2: tensor<77xi1>) -> (tensor<77xi1>, tensor<74x42xi32>) {
    %0 = tosa.argmax %arg0 {axis = 2 : i32} : (tensor<67x74x64x42xf32>) -> tensor<67x74x42xi32>
    %1 = tosa.logical_or %arg1, %arg2 : (tensor<77xi1>, tensor<77xi1>) -> tensor<77xi1>
    %2 = tosa.maximum %0, %0 : (tensor<67x74x42xi32>, tensor<67x74x42xi32>) -> tensor<67x74x42xi32>
    %3 = tosa.minimum %2, %2 : (tensor<67x74x42xi32>, tensor<67x74x42xi32>) -> tensor<67x74x42xi32>
    %4 = tosa.abs %3 : (tensor<67x74x42xi32>) -> tensor<67x74x42xi32>
    %5 = tosa.argmax %4 {axis = 0 : i32} : (tensor<67x74x42xi32>) -> tensor<74x42xi32>
    return %1, %5 : tensor<77xi1>, tensor<74x42xi32>
  }
}
