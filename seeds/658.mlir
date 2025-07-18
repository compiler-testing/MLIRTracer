module {
  func.func @main(%arg0: tensor<6x14x34x29x33xi8>, %arg1: tensor<6x14x34x29x1xi8>) -> tensor<6x14x34x58x33xi8> {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<6x14x34x29x33xi8>, tensor<6x14x34x29x1xi8>) -> tensor<6x14x34x29x33xi8>
    %1 = tosa.bitwise_or %0, %0 : (tensor<6x14x34x29x33xi8>, tensor<6x14x34x29x33xi8>) -> tensor<6x14x34x29x33xi8>
    %2 = tosa.clz %1 : (tensor<6x14x34x29x33xi8>) -> tensor<6x14x34x29x33xi8>
    %3 = tosa.concat %2, %0 {axis = 3 : i32} : (tensor<6x14x34x29x33xi8>, tensor<6x14x34x29x33xi8>) -> tensor<6x14x34x58x33xi8>
    return %3 : tensor<6x14x34x58x33xi8>
  }
}
