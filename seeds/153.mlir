module {
  func.func @main(%arg0: tensor<5x40x81x50x61xi1>, %arg1: tensor<5x1x1x1x61xi1>, %arg2: tensor<79x31x14xi8>) -> (tensor<5x40x81x50x61xi1>, tensor<1x31x14xi8>) {
    %0 = tosa.logical_or %arg0, %arg1 : (tensor<5x40x81x50x61xi1>, tensor<5x1x1x1x61xi1>) -> tensor<5x40x81x50x61xi1>
    %1 = tosa.reduce_min %arg2 {axis = 0 : i32} : (tensor<79x31x14xi8>) -> tensor<1x31x14xi8>
    %2 = tosa.reverse %1 {axis = 2 : i32} : (tensor<1x31x14xi8>) -> tensor<1x31x14xi8>
    %3 = tosa.maximum %2, %2 : (tensor<1x31x14xi8>, tensor<1x31x14xi8>) -> tensor<1x31x14xi8>
    return %0, %3 : tensor<5x40x81x50x61xi1>, tensor<1x31x14xi8>
  }
}
