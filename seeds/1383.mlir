module {
  func.func @main(%arg0: tensor<6xi1>) -> tensor<1xi1> {
    %0 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<6xi1>) -> tensor<1xi1>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %2 = tosa.reduce_any %1 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    return %2 : tensor<1xi1>
  }
}
