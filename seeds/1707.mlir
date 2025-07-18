module {
  func.func @main(%arg0: tensor<64xf32>, %arg1: tensor<i8>, %arg2: tensor<i8>, %arg3: tensor<65x83xi1>, %arg4: tensor<1x83xi1>) -> (tensor<i8>, tensor<1xf32>, tensor<65x83xi1>, tensor<65x83xi1>) {
    %0 = tosa.reciprocal %arg0 : (tensor<64xf32>) -> tensor<64xf32>
    %1 = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<64xf32>) -> tensor<1xf32>
    %2 = tosa.exp %1 : (tensor<1xf32>) -> tensor<1xf32>
    %3 = tosa.logical_right_shift %arg1, %arg2 : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %4 = tosa.logical_and %arg3, %arg4 : (tensor<65x83xi1>, tensor<1x83xi1>) -> tensor<65x83xi1>
    %5 = tosa.tanh %2 : (tensor<1xf32>) -> tensor<1xf32>
    %6 = tosa.reduce_product %5 {axis = 0 : i32} : (tensor<1xf32>) -> tensor<1xf32>
    %7 = tosa.clz %4 : (tensor<65x83xi1>) -> tensor<65x83xi1>
    %8 = tosa.clz %4 : (tensor<65x83xi1>) -> tensor<65x83xi1>
    return %3, %6, %7, %8 : tensor<i8>, tensor<1xf32>, tensor<65x83xi1>, tensor<65x83xi1>
  }
}
