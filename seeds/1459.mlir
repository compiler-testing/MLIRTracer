module {
  func.func @main(%arg0: tensor<13x74xi16>, %arg1: tensor<39xi1>) -> (tensor<13x1xi16>, tensor<1xi1>) {
    %0 = tosa.reduce_product %arg0 {axis = 1 : i32} : (tensor<13x74xi16>) -> tensor<13x1xi16>
    %1 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<39xi1>) -> tensor<1xi1>
    %2 = tosa.add %1, %1 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %0, %2 : tensor<13x1xi16>, tensor<1xi1>
  }
}
