module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<63xi32>, %arg3: tensor<1xi32>, %arg4: tensor<13xi1>) -> (tensor<i1>, tensor<1xi32>, tensor<63xi32>, tensor<1xi1>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.intdiv %arg2, %arg3 : (tensor<63xi32>, tensor<1xi32>) -> tensor<63xi32>
    %2 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<63xi32>) -> tensor<1xi32>
    %3 = tosa.minimum %1, %1 : (tensor<63xi32>, tensor<63xi32>) -> tensor<63xi32>
    %4 = tosa.intdiv %2, %2 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %5 = tosa.intdiv %3, %1 : (tensor<63xi32>, tensor<63xi32>) -> tensor<63xi32>
    %6 = tosa.add %5, %3 : (tensor<63xi32>, tensor<63xi32>) -> tensor<63xi32>
    %7 = tosa.add %6, %5 : (tensor<63xi32>, tensor<63xi32>) -> tensor<63xi32>
    %8 = tosa.reduce_all %arg4 {axis = 0 : i32} : (tensor<13xi1>) -> tensor<1xi1>
    return %0, %4, %7, %8 : tensor<i1>, tensor<1xi32>, tensor<63xi32>, tensor<1xi1>
  }
}
