module {
  func.func @main(%arg0: tensor<43x99x78x98xi16>, %arg1: tensor<83xi1>) -> (tensor<1x99x78x98xi16>, tensor<1xi1>) {
    %0 = tosa.reduce_product %arg0 {axis = 0 : i32} : (tensor<43x99x78x98xi16>) -> tensor<1x99x78x98xi16>
    %1 = tosa.sub %0, %0 : (tensor<1x99x78x98xi16>, tensor<1x99x78x98xi16>) -> tensor<1x99x78x98xi16>
    %2 = tosa.reduce_all %arg1 {axis = 0 : i32} : (tensor<83xi1>) -> tensor<1xi1>
    %3 = tosa.logical_and %2, %2 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %1, %3 : tensor<1x99x78x98xi16>, tensor<1xi1>
  }
}
