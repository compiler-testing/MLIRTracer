module {
  func.func @main(%arg0: tensor<3x89xi8>, %arg1: tensor<3x6xi8>, %arg2: tensor<80xi1>) -> (tensor<1xi1>, tensor<6x1xi8>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<3x89xi8>, tensor<3x6xi8>) -> tensor<3x95xi8>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<3x95xi8>, tensor<3x95xi8>) -> tensor<6x95xi8>
    %2 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<80xi1>) -> tensor<1xi1>
    %3 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<6x95xi8>) -> tensor<6x1xi8>
    return %2, %3 : tensor<1xi1>, tensor<6x1xi8>
  }
}
