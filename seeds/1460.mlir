module {
  func.func @main(%arg0: tensor<8x8x80x93xi8>, %arg1: tensor<83x33x48x43x79x67xf32>) -> (tensor<8x1x80x93xi1>, tensor<83x33x48x43x79x67xf32>, tensor<83x33x48x43x79x67xf32>) {
    %0 = tosa.abs %arg0 : (tensor<8x8x80x93xi8>) -> tensor<8x8x80x93xi8>
    %1 = tosa.add %0, %0 : (tensor<8x8x80x93xi8>, tensor<8x8x80x93xi8>) -> tensor<8x8x80x93xi8>
    %2 = tosa.maximum %1, %1 : (tensor<8x8x80x93xi8>, tensor<8x8x80x93xi8>) -> tensor<8x8x80x93xi8>
    %3 = tosa.equal %2, %1 : (tensor<8x8x80x93xi8>, tensor<8x8x80x93xi8>) -> tensor<8x8x80x93xi1>
    %4 = tosa.bitwise_and %3, %3 : (tensor<8x8x80x93xi1>, tensor<8x8x80x93xi1>) -> tensor<8x8x80x93xi1>
    %5 = tosa.bitwise_xor %4, %3 : (tensor<8x8x80x93xi1>, tensor<8x8x80x93xi1>) -> tensor<8x8x80x93xi1>
    %6 = tosa.reduce_all %5 {axis = 1 : i32} : (tensor<8x8x80x93xi1>) -> tensor<8x1x80x93xi1>
    %7 = tosa.reciprocal %arg1 : (tensor<83x33x48x43x79x67xf32>) -> tensor<83x33x48x43x79x67xf32>
    %8 = tosa.sigmoid %7 : (tensor<83x33x48x43x79x67xf32>) -> tensor<83x33x48x43x79x67xf32>
    %9 = tosa.sigmoid %7 : (tensor<83x33x48x43x79x67xf32>) -> tensor<83x33x48x43x79x67xf32>
    return %6, %8, %9 : tensor<8x1x80x93xi1>, tensor<83x33x48x43x79x67xf32>, tensor<83x33x48x43x79x67xf32>
  }
}
