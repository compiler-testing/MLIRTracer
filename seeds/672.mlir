module {
  func.func @main(%arg0: tensor<90x16x78x69xf32>, %arg1: tensor<i1>, %arg2: tensor<i1>) -> (tensor<i1>, tensor<1x16x78x69xi1>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<90x16x78x69xf32>) -> tensor<1x16x78x69xf32>
    %1 = tosa.maximum %0, %0 : (tensor<1x16x78x69xf32>, tensor<1x16x78x69xf32>) -> tensor<1x16x78x69xf32>
    %2 = tosa.reverse %1 {axis = 3 : i32} : (tensor<1x16x78x69xf32>) -> tensor<1x16x78x69xf32>
    %3 = tosa.logical_and %arg1, %arg2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.floor %2 : (tensor<1x16x78x69xf32>) -> tensor<1x16x78x69xf32>
    %5 = tosa.pow %4, %4 : (tensor<1x16x78x69xf32>, tensor<1x16x78x69xf32>) -> tensor<1x16x78x69xf32>
    %6 = tosa.reduce_product %5 {axis = 0 : i32} : (tensor<1x16x78x69xf32>) -> tensor<1x16x78x69xf32>
    %7 = tosa.equal %6, %0 : (tensor<1x16x78x69xf32>, tensor<1x16x78x69xf32>) -> tensor<1x16x78x69xi1>
    %8 = tosa.logical_and %7, %7 : (tensor<1x16x78x69xi1>, tensor<1x16x78x69xi1>) -> tensor<1x16x78x69xi1>
    return %3, %8 : tensor<i1>, tensor<1x16x78x69xi1>
  }
}
