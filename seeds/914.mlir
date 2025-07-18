module {
  func.func @main(%arg0: tensor<56xi32>, %arg1: tensor<90xi32>, %arg2: tensor<i1>, %arg3: tensor<75xf32>) -> (tensor<146xi32>, tensor<75xf32>, tensor<i1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<56xi32>, tensor<90xi32>) -> tensor<146xi32>
    %1 = tosa.bitwise_and %0, %0 : (tensor<146xi32>, tensor<146xi32>) -> tensor<146xi32>
    %2 = tosa.logical_not %arg2 : (tensor<i1>) -> tensor<i1>
    %3 = tosa.reverse %1 {axis = 0 : i32} : (tensor<146xi32>) -> tensor<146xi32>
    %4 = tosa.exp %arg3 : (tensor<75xf32>) -> tensor<75xf32>
    %5 = tosa.exp %4 : (tensor<75xf32>) -> tensor<75xf32>
    %6 = tosa.logical_not %2 : (tensor<i1>) -> tensor<i1>
    return %3, %5, %6 : tensor<146xi32>, tensor<75xf32>, tensor<i1>
  }
}
