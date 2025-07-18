module {
  func.func @main(%arg0: tensor<12x35xi8>, %arg1: tensor<70x63x5x100x51x67xf32>, %arg2: tensor<1x63x5x1x1x67xf32>) -> (tensor<1x35xi8>, tensor<70x63x5x100x51x67xf32>) {
    %0 = tosa.identity %arg0 : (tensor<12x35xi8>) -> tensor<12x35xi8>
    %1 = tosa.add %0, %0 : (tensor<12x35xi8>, tensor<12x35xi8>) -> tensor<12x35xi8>
    %2 = tosa.logical_left_shift %1, %0 : (tensor<12x35xi8>, tensor<12x35xi8>) -> tensor<12x35xi8>
    %3 = tosa.reduce_product %2 {axis = 0 : i32} : (tensor<12x35xi8>) -> tensor<1x35xi8>
    %4 = tosa.pow %arg1, %arg2 : (tensor<70x63x5x100x51x67xf32>, tensor<1x63x5x1x1x67xf32>) -> tensor<70x63x5x100x51x67xf32>
    return %3, %4 : tensor<1x35xi8>, tensor<70x63x5x100x51x67xf32>
  }
}
