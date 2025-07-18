module {
  func.func @main(%arg0: tensor<20x53x47x44x34x23xf32>, %arg1: tensor<69x79x68xi1>, %arg2: tensor<1x1x68xi1>) -> (tensor<20x53x47x44x34x23xf32>, tensor<1x1x68xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<20x53x47x44x34x23xf32>) -> tensor<20x53x47x44x34x23xf32>
    %1 = tosa.logical_right_shift %arg1, %arg2 : (tensor<69x79x68xi1>, tensor<1x1x68xi1>) -> tensor<69x79x68xi1>
    %2 = tosa.pow %0, %0 : (tensor<20x53x47x44x34x23xf32>, tensor<20x53x47x44x34x23xf32>) -> tensor<20x53x47x44x34x23xf32>
    %3 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<69x79x68xi1>) -> tensor<69x1x68xi1>
    %4 = tosa.reduce_any %3 {axis = 0 : i32} : (tensor<69x1x68xi1>) -> tensor<1x1x68xi1>
    return %2, %4 : tensor<20x53x47x44x34x23xf32>, tensor<1x1x68xi1>
  }
}
