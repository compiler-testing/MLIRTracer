module {
  func.func @main(%arg0: tensor<26x52x42xi1>, %arg1: tensor<87xf32>) -> (tensor<26x1x42xi1>, tensor<87xf32>) {
    %0 = tosa.reduce_any %arg0 {axis = 1 : i32} : (tensor<26x52x42xi1>) -> tensor<26x1x42xi1>
    %1 = tosa.floor %arg1 : (tensor<87xf32>) -> tensor<87xf32>
    return %0, %1 : tensor<26x1x42xi1>, tensor<87xf32>
  }
}
