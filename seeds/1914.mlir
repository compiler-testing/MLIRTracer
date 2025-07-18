module {
  func.func @main(%arg0: tensor<2x84xi16>, %arg1: tensor<4x96x41x82x61xi1>, %arg2: tensor<1x1x1x1x1xi1>, %arg3: tensor<62x53x19x82xi1>) -> (tensor<2x84xi16>, tensor<4x96x41x82x61xi1>, tensor<62x53x1x82xi1>) {
    %0 = tosa.reverse %arg0 {axis = 1 : i32} : (tensor<2x84xi16>) -> tensor<2x84xi16>
    %1 = tosa.reverse %0 {axis = 1 : i32} : (tensor<2x84xi16>) -> tensor<2x84xi16>
    %2 = tosa.logical_xor %arg1, %arg2 : (tensor<4x96x41x82x61xi1>, tensor<1x1x1x1x1xi1>) -> tensor<4x96x41x82x61xi1>
    %3 = tosa.reduce_any %arg3 {axis = 2 : i32} : (tensor<62x53x19x82xi1>) -> tensor<62x53x1x82xi1>
    %4 = tosa.reduce_sum %3 {axis = 2 : i32} : (tensor<62x53x1x82xi1>) -> tensor<62x53x1x82xi1>
    %5 = tosa.logical_xor %4, %3 : (tensor<62x53x1x82xi1>, tensor<62x53x1x82xi1>) -> tensor<62x53x1x82xi1>
    return %1, %2, %5 : tensor<2x84xi16>, tensor<4x96x41x82x61xi1>, tensor<62x53x1x82xi1>
  }
}
