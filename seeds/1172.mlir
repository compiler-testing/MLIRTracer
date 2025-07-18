module {
  func.func @main(%arg0: tensor<78x60x78x31x26xi8>, %arg1: tensor<78x1x1x1x1xi8>, %arg2: tensor<85x82xf32>) -> (tensor<78x60x78x31x26xi8>, tensor<82xi32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<78x60x78x31x26xi8>, tensor<78x1x1x1x1xi8>) -> tensor<78x60x78x31x26xi8>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<78x60x78x31x26xi8>, tensor<78x60x78x31x26xi8>) -> tensor<78x60x78x31x26xi8>
    %2 = tosa.minimum %1, %1 : (tensor<78x60x78x31x26xi8>, tensor<78x60x78x31x26xi8>) -> tensor<78x60x78x31x26xi8>
    %3 = tosa.add %2, %2 : (tensor<78x60x78x31x26xi8>, tensor<78x60x78x31x26xi8>) -> tensor<78x60x78x31x26xi8>
    %4 = tosa.sigmoid %arg2 : (tensor<85x82xf32>) -> tensor<85x82xf32>
    %5 = tosa.argmax %4 {axis = 0 : i32} : (tensor<85x82xf32>) -> tensor<82xi32>
    return %3, %5 : tensor<78x60x78x31x26xi8>, tensor<82xi32>
  }
}
