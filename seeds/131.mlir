module {
  func.func @main(%arg0: tensor<77xi1>, %arg1: tensor<42xf32>) -> (tensor<2xi1>, tensor<42xi1>) {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<77xi1>) -> tensor<1xi1>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<1xi1>, tensor<1xi1>) -> tensor<2xi1>
    %2 = tosa.sigmoid %arg1 : (tensor<42xf32>) -> tensor<42xf32>
    %3 = tosa.greater_equal %2, %2 : (tensor<42xf32>, tensor<42xf32>) -> tensor<42xi1>
    return %1, %3 : tensor<2xi1>, tensor<42xi1>
  }
}
