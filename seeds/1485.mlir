module {
  func.func @main(%arg0: tensor<44xi8>) -> tensor<2xi8> {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<44xi8>) -> tensor<1xi8>
    %1 = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<1xi8>) -> tensor<1xi8>
    %2 = tosa.concat %1, %1 {axis = 0 : i32} : (tensor<1xi8>, tensor<1xi8>) -> tensor<2xi8>
    return %2 : tensor<2xi8>
  }
}
