module {
  func.func @main(%arg0: tensor<76x80xi1>) -> tensor<76x2xi1> {
    %0 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<76x80xi1>) -> tensor<76x1xi1>
    %1 = tosa.reduce_any %0 {axis = 1 : i32} : (tensor<76x1xi1>) -> tensor<76x1xi1>
    %2 = tosa.reduce_all %1 {axis = 1 : i32} : (tensor<76x1xi1>) -> tensor<76x1xi1>
    %3 = tosa.bitwise_and %2, %0 : (tensor<76x1xi1>, tensor<76x1xi1>) -> tensor<76x1xi1>
    %4 = tosa.concat %3, %0 {axis = 1 : i32} : (tensor<76x1xi1>, tensor<76x1xi1>) -> tensor<76x2xi1>
    %5 = tosa.logical_or %4, %4 : (tensor<76x2xi1>, tensor<76x2xi1>) -> tensor<76x2xi1>
    return %5 : tensor<76x2xi1>
  }
}
