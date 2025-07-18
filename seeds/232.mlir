module {
  func.func @main(%arg0: tensor<97x66x31x80x83xi8>, %arg1: tensor<i1>) -> (tensor<1x12x4x12x4xi8>, tensor<i1>) {
    %0 = tosa.identity %arg0 : (tensor<97x66x31x80x83xi8>) -> tensor<97x66x31x80x83xi8>
    %s_1_start = tosa.const_shape {values = dense<[ 68, 39, 27, 68, 47 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_1_size = tosa.const_shape {values = dense<[ 1, 6, 4, 12, 4 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %1 = tosa.slice %0, %s_1_start, %s_1_size : (tensor<97x66x31x80x83xi8>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<1x6x4x12x4xi8>
    %2 = tosa.concat %1, %1 {axis = 1 : i32} : (tensor<1x6x4x12x4xi8>, tensor<1x6x4x12x4xi8>) -> tensor<1x12x4x12x4xi8>
    %3 = tosa.maximum %2, %2 : (tensor<1x12x4x12x4xi8>, tensor<1x12x4x12x4xi8>) -> tensor<1x12x4x12x4xi8>
    %4 = tosa.logical_not %arg1 : (tensor<i1>) -> tensor<i1>
    return %3, %4 : tensor<1x12x4x12x4xi8>, tensor<i1>
  }
}
